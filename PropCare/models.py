from abc import ABC
import tensorflow as tf
import sys
from tensorflow import keras
from tensorflow.keras.losses import MSE, binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Flatten, Dense, BatchNormalization
from tensorflow.keras.saving import register_keras_serializable
import tensorflow_probability as tfp
# from utils import squared_dist, find_k_smallest

@register_keras_serializable(package="Causal_Model")
class Causal_Model(Model, ABC):
    def __init__(self, num_users, num_items, flags, user_embs, item_embs, item_popularity,
                 **kwargs):
        super(Causal_Model, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.flags = flags
        self.item_popularity = tf.cast(tf.squeeze(item_popularity), tf.float32)
        self.estimator_layer_units = flags.estimator_layer_units
        self.click_layer_units = flags.click_layer_units
        self.emb_layer_units = flags.embedding_layer_units
        self.lambda_1 = flags.lambda_1
        self.dims = flags.dimension
        self.p_weight = flags.p_weight
        self.norm_layer = tf.keras.constraints.non_neg()
        if user_embs is None:
            self.mf_user_embedding = Embedding(input_dim=num_users, output_dim=flags.dimension,
                                               name='mf_user_embedding', trainable=False,
                                               embeddings_regularizer="l2")
        else:
            self.mf_user_embedding = Embedding(input_dim=num_users, output_dim=flags.dimension,
                                               name='mf_user_embedding', weights=[user_embs],
                                               trainable=False, embeddings_regularizer="l2")
        if item_embs is None:
            self.mf_item_embedding = Embedding(input_dim=num_items, output_dim=flags.dimension,
                                               name='mf_item_embedding', trainable=False,
                                               embeddings_regularizer="l2")
        else:
            self.mf_item_embedding = Embedding(input_dim=num_items, output_dim=flags.dimension,
                                               name='mf_item_embedding', weights=[item_embs],
                                               trainable=False, embeddings_regularizer="l2")
        self.flatten_layers = Flatten()
        self.emb_layers = []
        for i, unit in enumerate(flags.embedding_layer_units):
            self.emb_layers.append(
                Dense(unit, activation=tf.keras.layers.LeakyReLU(), name="emb_{}".format(i), kernel_initializer='he_normal', trainable=True))
        self.propensity_layers = []
        self.relevance_layers = []
        self.propensity_bn_layers = []
        self.relevance_bn_layers = []
        self.film_alpha_propensity = []
        self.film_beta_propensity = []
        self.film_alpha_relevance = []
        self.film_beta_relevance = []
        self.exp_weight = tf.Variable(1.0, trainable=True)
        for i, unit in enumerate(flags.estimator_layer_units):
            self.film_alpha_propensity.append(
                Dense(unit, activation=tf.keras.layers.LeakyReLU(), name='film_alpha_propensity_{}'.format(i),
                      kernel_initializer="he_normal", trainable=True))
            self.film_beta_propensity.append(
                Dense(unit, activation=tf.keras.layers.LeakyReLU(), name='film_beta_propensity_{}'.format(i),
                      kernel_initializer="he_normal", trainable=True))
            self.film_alpha_relevance.append(
                Dense(unit, activation=tf.keras.layers.LeakyReLU(), name='film_alpha_relevance_{}'.format(i),
                      kernel_initializer="he_normal", trainable=True))
            self.film_beta_relevance.append(
                Dense(unit, activation=tf.keras.layers.LeakyReLU(), name='film_beta_relevance_{}'.format(i), 
                      kernel_initializer="he_normal", trainable=True))
            self.propensity_bn_layers.append(
                BatchNormalization(name='batch_norm_propensity_{}'.format(i), trainable=True))
            self.relevance_bn_layers.append(
                BatchNormalization(name='batch_norm_relevance_{}'.format(i), trainable=True))
            self.propensity_layers.append(
                Dense(unit, activation=tf.keras.layers.LeakyReLU(), name='propensity_{}'.format(i), kernel_regularizer="l2",
                      kernel_initializer="he_normal", trainable=True))
            self.relevance_layers.append(
                Dense(unit, activation=tf.keras.layers.LeakyReLU(), name='relevance_{}'.format(i), kernel_regularizer="l2",
                      kernel_initializer="he_normal", trainable=True))
        self.propensity_Prediction_layer = Dense(1, activation='sigmoid',
                                                 name="propensity_prediction", kernel_regularizer="l2",
                                                 kernel_initializer="he_normal", trainable=True)
        self.relevance_Prediction_layer = Dense(1, activation='sigmoid',
                                                name="relevance_prediction", kernel_regularizer="l2",
                                                kernel_initializer="he_normal", trainable=True)
        self.kl = tf.keras.losses.KLDivergence()
        self.target_dist = tfp.distributions.Beta(0.2, 1.0)
        self.estimator_optimizer = keras.optimizers.SGD(keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=0.01, first_decay_steps=2000))
    
    def get_config(self):
        """Safe serialization with proper numpy array handling"""
        config = super().get_config()
        
        # Required attributes
        config.update({
            'num_users': int(self.num_users),
            'num_items': int(self.num_items),
            'flags': {
                'estimator_layer_units': list(self.flags.estimator_layer_units),
                'click_layer_units': list(self.flags.click_layer_units),
                'embedding_layer_units': list(self.flags.embedding_layer_units),
                'lambda_1': float(self.flags.lambda_1),
                'dimension': int(self.flags.dimension),
                'p_weight': float(self.flags.p_weight)
            },
            'model_version': '1.3'
        })

        # Handle numpy arrays safely
        def serialize_array(arr):
            if arr is None:
                return None
            if isinstance(arr, (np.ndarray, tf.Tensor)):
                return {'__numpy__': True, 'value': arr.tolist(), 'dtype': str(arr.dtype)}
            return arr

        # Optional attributes
        config.update({
            'user_embs': serialize_array(self.mf_user_embedding.weights[0] if self.mf_user_embedding.weights else None),
            'item_embs': serialize_array(self.mf_item_embedding.weights[0] if self.mf_item_embedding.weights else None),
            'item_popularity': serialize_array(self.item_popularity.numpy() if hasattr(self, 'item_popularity') else None)
        })

        return config

    @classmethod
    def from_config(cls, config):
        """Safe deserialization with proper tensor reconstruction"""
        # Validate config version
        version = config.get('model_version', '1.0')
        if version not in ['1.0', '1.1', '1.2', '1.3']:
            raise ValueError(f"Unsupported model version: {version}")

        # Convert serialized arrays back to tensors
        def deserialize_array(data):
            if data is None:
                return None
            if isinstance(data, dict) and '__numpy__' in data:
                return tf.constant(data['value'], dtype=data['dtype'])
            if isinstance(data, list):
                return tf.constant(data)
            return data

        # Process required parameters
        try:
            num_users = int(config['num_users'])
            num_items = int(config['num_items'])
            
            # Reconstruct flags
            flags_config = config['flags']
            class Flags:
                def __init__(self, config):
                    self.estimator_layer_units = list(config['estimator_layer_units'])
                    self.click_layer_units = list(config['click_layer_units'])
                    self.embedding_layer_units = list(config['embedding_layer_units'])
                    self.lambda_1 = float(config['lambda_1'])
                    self.dimension = int(config['dimension'])
                    self.p_weight = float(config['p_weight'])
            
            flags = Flags(flags_config)

            # Process tensors with proper error handling
            item_popularity = deserialize_array(config.get('item_popularity'))
            if item_popularity is None:
                item_popularity = tf.zeros((num_items,), dtype=tf.float32)
            elif len(item_popularity.shape) == 0:
                item_popularity = tf.zeros((num_items,), dtype=tf.float32)

            user_embs = deserialize_array(config.get('user_embs'))
            item_embs = deserialize_array(config.get('item_embs'))

            return cls(
                num_users=num_users,
                num_items=num_items,
                flags=flags,
                user_embs=user_embs,
                item_embs=item_embs,
                item_popularity=item_popularity
            )
            
        except KeyError as e:
            raise ValueError(f"Missing required config key: {str(e)}") from e
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid config value: {str(e)}") from e

    @tf.function()
    def call(self, inputs, training=None, **kwargs):
        user_input, item_input = inputs
        mf_user_latent = self.flatten_layers(self.norm_layer(self.mf_user_embedding(user_input)))
        mf_item_latent = self.flatten_layers(self.norm_layer(self.mf_item_embedding(item_input)))
        mf_vector = tf.concat((mf_user_latent, mf_item_latent), axis=1)
        for i, unit in enumerate(self.emb_layer_units):
            emb_layer = self.emb_layers[i]
            mf_vector = emb_layer(mf_vector)
        propensity_vector = mf_vector
        relevance_vector = mf_vector
        film_reg_loss = 0.0
        for i, unit in enumerate(self.estimator_layer_units):
            propensity_layer = self.propensity_layers[i]
            film_alpha_propensity_layer = self.film_alpha_propensity[i]
            film_beta_propensity_layer = self.film_beta_propensity[i]
            propensity_bn_layer = self.propensity_bn_layers[i]
            propensity_vector = propensity_layer(propensity_vector)
            film_alpha_propensity = film_alpha_propensity_layer(mf_vector)
            film_beta_propensity = film_beta_propensity_layer(mf_vector)
            film_reg_loss += tf.nn.l2_loss(film_alpha_propensity - 1)
            film_reg_loss += tf.nn.l2_loss(film_beta_propensity)
            propensity_vector = tf.nn.leaky_relu(
                tf.multiply(propensity_vector, film_alpha_propensity) + film_beta_propensity)
            propensity_vector = propensity_bn_layer(propensity_vector, training=training)
            relevance_layer = self.relevance_layers[i]
            film_alpha_relevance_layer = self.film_alpha_relevance[i]
            film_beta_relevance_layer = self.film_beta_relevance[i]
            relevance_bn_layer = self.relevance_bn_layers[i]
            relevance_vector = relevance_layer(relevance_vector)
            film_alpha_relevance = film_alpha_relevance_layer(mf_vector)
            film_beta_relevance = film_beta_relevance_layer(mf_vector)
            film_reg_loss += tf.nn.l2_loss(film_alpha_relevance - 1)
            film_reg_loss += tf.nn.l2_loss(film_beta_relevance)
            relevance_vector = tf.nn.leaky_relu(
                tf.multiply(relevance_vector, film_alpha_relevance) + film_beta_relevance)
            relevance_vector = relevance_bn_layer(relevance_vector, training=training)
        propensity = self.propensity_Prediction_layer(propensity_vector)
        relevance = self.relevance_Prediction_layer(relevance_vector)
        propensity = tf.reshape(propensity, [-1, 1])
        relevance = tf.reshape(relevance, [-1, 1])
        propensity = tf.clip_by_value(propensity, 0.0001, 0.9999)
        relevance = tf.clip_by_value(relevance, 0.0001, 0.9999)
        click = tf.multiply(propensity, relevance)
        return click, propensity, relevance, film_reg_loss
    
    @tf.function()
    def propensity_train(self, data):
        user, item_i, item_j, y_true = data
        y_true = tf.reshape(y_true, [-1, 1]) 
        user = tf.reshape(user, [-1, 1])
        item_i = tf.reshape(item_i, [-1, 1])
        item_j = tf.reshape(item_j, [-1, 1])

        with tf.GradientTape() as tape2:
            y_i, p_i, r_i, film_reg_loss_1 = self((user, item_i), training=True)
            y_j, p_j, r_j, film_reg_loss_2 = self((user, item_j), training=True)
            
            # Calculating different loss components
            loss_click = tf.reduce_mean(binary_crossentropy(y_true=y_true, y_pred=y_i))
            pop_signs = tf.sign(tf.gather(self.item_popularity, tf.squeeze(item_i)) - tf.gather(self.item_popularity, tf.squeeze(item_j)))
            pop_signs = tf.reshape(pop_signs, [-1, 1])
            
            # Pairwise losses and regularization
            p_diff = tf.multiply(pop_signs, (p_i - p_j))
            r_diff = tf.multiply(pop_signs, (r_j - r_i))
            y_diff = tf.multiply(pop_signs, (y_i - y_j))
            
            weights_loss = tf.exp(-self.exp_weight * tf.square(y_diff))
            weights_loss = weights_loss / tf.math.reduce_max(weights_loss)
            loss_pair = tf.math.log(tf.math.sigmoid(p_diff) + tf.math.sigmoid(r_diff))
            loss_pair = tf.multiply(weights_loss, loss_pair)
            loss_pair = -tf.reduce_mean(loss_pair)
            
            # KL divergence regularization
            target_samples_i = tf.stop_gradient(self.target_dist.sample(tf.shape(p_i)))
            target_samples_j = tf.stop_gradient(self.target_dist.sample(tf.shape(p_j)))
            
            q1 = tf.clip_by_value(tf.sort(target_samples_i, axis=0), 0.0001, 0.9999)
            q2 = tf.clip_by_value(tf.sort(target_samples_j, axis=0), 0.0001, 0.9999)
            p1 = tf.clip_by_value(tf.sort(p_i, axis=0), 0.0001, 0.9999)
            p2 = tf.clip_by_value(tf.sort(p_j, axis=0), 0.0001, 0.9999)

            p_loss = self.kl(p1, q1) + self.kl(p2, q2)
            reg_loss = 0.0001 * (tf.add_n(self.losses) + film_reg_loss_1 + film_reg_loss_2) + self.p_weight * p_loss

            # Final loss
            loss = self.lambda_1 * loss_pair + loss_click + reg_loss
        
        # Calculate gradients
        gradients = tape2.gradient(loss, self.trainable_weights)

        # Apply gradients using the optimizer
        self.estimator_optimizer.apply_gradients(zip(gradients, self.trainable_weights))

if __name__ == "__main__":
    pass

