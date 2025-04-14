from abc import ABC
import tensorflow as tf
import sys
from tensorflow import keras
from tensorflow.keras.losses import MSE, binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Flatten, Dense, BatchNormalization
import tensorflow_probability as tfp
# from utils import squared_dist, find_k_smallest

class Causal_Model(Model, ABC):
    def __init__(self, num_users, num_items, flags, user_embs, item_embs, item_popularity, **kwargs):
        super(Causal_Model, self).__init__(**kwargs)
        self.item_popularity = tf.cast(tf.squeeze(item_popularity), tf.float32)
        self.estimator_layer_units = flags.estimator_layer_units
        self.click_layer_units = flags.click_layer_units
        self.emb_layer_units = flags.embedding_layer_units
        self.lambda_1 = flags.lambda_1
        self.lambda_2 = flags.lambda_2
        self.lambda_3 = flags.lambda_3
        self.dims = flags.dimension
        self.p_weight = flags.p_weight
        self.r_weight = flags.r_weight
        self.norm_layer = tf.keras.constraints.non_neg()

        # Эмбеддинги
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
        self.emb_layers = [Dense(unit, activation=tf.keras.layers.LeakyReLU(), 
                                 name=f"emb_{i}", kernel_initializer='he_normal', trainable=True)
                           for i, unit in enumerate(flags.embedding_layer_units)]

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
            self.film_alpha_propensity.append(Dense(unit, activation=tf.keras.layers.LeakyReLU(),
                                                    name=f'film_alpha_propensity_{i}',
                                                    kernel_initializer="he_normal", trainable=True))
            self.film_beta_propensity.append(Dense(unit, activation=tf.keras.layers.LeakyReLU(),
                                                   name=f'film_beta_propensity_{i}',
                                                   kernel_initializer="he_normal", trainable=True))
            self.film_alpha_relevance.append(Dense(unit, activation=tf.keras.layers.LeakyReLU(),
                                                   name=f'film_alpha_relevance_{i}',
                                                   kernel_initializer="he_normal", trainable=True))
            self.film_beta_relevance.append(Dense(unit, activation=tf.keras.layers.LeakyReLU(),
                                                  name=f'film_beta_relevance_{i}',
                                                  kernel_initializer="he_normal", trainable=True))
            self.propensity_layers.append(Dense(unit, activation=tf.keras.layers.LeakyReLU(),
                                                name=f'propensity_{i}', kernel_regularizer="l2",
                                                kernel_initializer="he_normal", trainable=True))
            self.relevance_layers.append(Dense(unit, activation=tf.keras.layers.LeakyReLU(),
                                               name=f'relevance_{i}', kernel_regularizer="l2",
                                               kernel_initializer="he_normal", trainable=True))
            self.propensity_bn_layers.append(BatchNormalization(name=f'batch_norm_propensity_{i}', trainable=True))
            self.relevance_bn_layers.append(BatchNormalization(name=f'batch_norm_relevance_{i}', trainable=True))

        # LayerNorm и Dropout
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.relevance_dropout = tf.keras.layers.Dropout(0.3)

        self.propensity_Prediction_layer = Dense(1, activation='sigmoid',
                                                 name="propensity_prediction", kernel_regularizer="l2",
                                                 kernel_initializer="he_normal", trainable=True)
        self.relevance_Prediction_layer = Dense(1, activation='sigmoid',
                                                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                                                bias_initializer=tf.keras.initializers.Constant(0.0),
                                                name="relevance_prediction", kernel_regularizer="l2", trainable=True)

        self.kl = tf.keras.losses.KLDivergence()
        self.target_dist = tfp.distributions.Beta(0.2, 1.0)
        self.estimator_optimizer = keras.optimizers.SGD(
            keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=0.01, first_decay_steps=2000)
        )

    @tf.function()
    def call(self, inputs, training=None, **kwargs):
        user_input, item_input = inputs
        mf_user_latent = self.flatten_layers(self.norm_layer(self.mf_user_embedding(user_input)))
        mf_item_latent = self.flatten_layers(self.norm_layer(self.mf_item_embedding(item_input)))
        mf_vector = tf.concat((mf_user_latent, mf_item_latent), axis=1)

        mf_vector = self.layer_norm(mf_vector)

        for emb_layer in self.emb_layers:
            mf_vector = emb_layer(mf_vector)

        propensity_vector = mf_vector
        relevance_vector = mf_vector
        film_reg_loss = 0.0

        for i in range(len(self.estimator_layer_units)):
            # Propensity
            propensity_vector = self.propensity_layers[i](propensity_vector)
            film_alpha_p = self.film_alpha_propensity[i](mf_vector)
            film_beta_p = self.film_beta_propensity[i](mf_vector)
            film_reg_loss += tf.nn.l2_loss(film_alpha_p - 1)
            film_reg_loss += tf.nn.l2_loss(film_beta_p)
            propensity_vector = tf.nn.leaky_relu(propensity_vector * film_alpha_p + film_beta_p)
            propensity_vector = self.propensity_bn_layers[i](propensity_vector, training=training)

            # Relevance
            relevance_vector = self.relevance_layers[i](relevance_vector)
            film_alpha_r = self.film_alpha_relevance[i](mf_vector)
            film_beta_r = self.film_beta_relevance[i](mf_vector)
            film_reg_loss += tf.nn.l2_loss(film_alpha_r - 1)
            film_reg_loss += tf.nn.l2_loss(film_beta_r)
            relevance_vector = tf.nn.leaky_relu(relevance_vector * film_alpha_r + film_beta_r)
            relevance_vector = self.relevance_dropout(relevance_vector, training=training)
            relevance_vector = self.relevance_bn_layers[i](relevance_vector, training=training)

        propensity = self.propensity_Prediction_layer(propensity_vector)
        relevance = self.relevance_Prediction_layer(relevance_vector)

        propensity = tf.reshape(propensity, [-1, 1])
        relevance = tf.reshape(relevance, [-1, 1])

        # Safety clipping
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

        with tf.GradientTape() as tape:
            y_i, p_i, r_i, film_reg_loss_1 = self((user, item_i), training=True)
            y_j, p_j, r_j, film_reg_loss_2 = self((user, item_j), training=True)

            loss_click = tf.reduce_mean(binary_crossentropy(y_true=y_true, y_pred=y_i))
            pop_signs = tf.sign(tf.gather(self.item_popularity, tf.squeeze(item_i)) - 
                                tf.gather(self.item_popularity, tf.squeeze(item_j)))
            pop_signs = tf.reshape(pop_signs, [-1, 1])

            p_diff = tf.multiply(pop_signs, (p_i - p_j))
            r_diff = tf.multiply(pop_signs, (r_j - r_i))
            y_diff = tf.multiply(pop_signs, (y_i - y_j))

            weights_loss = tf.exp(-self.exp_weight * tf.square(y_diff))
            weights_loss = weights_loss / tf.reduce_max(weights_loss)

            loss_pair = tf.math.log(tf.math.sigmoid(p_diff)) + tf.math.log(tf.math.sigmoid(r_diff))
            loss_pair = tf.multiply(weights_loss, loss_pair)
            loss_pair = -tf.reduce_mean(loss_pair)

            # KL divergence regularization
            target_p_samples_i = tf.stop_gradient(self.target_dist.sample(tf.shape(p_i)))
            target_p_samples_j = tf.stop_gradient(self.target_dist.sample(tf.shape(p_j)))

            qp1 = tf.clip_by_value(tf.sort(target_p_samples_i, axis=0), 0.0001, 0.9999)
            qp2 = tf.clip_by_value(tf.sort(target_p_samples_j, axis=0), 0.0001, 0.9999)
            p1 = tf.clip_by_value(tf.sort(p_i, axis=0), 0.0001, 0.9999)
            p2 = tf.clip_by_value(tf.sort(p_j, axis=0), 0.0001, 0.9999)

            p_loss = self.kl(p1, qp1) + self.kl(p2, qp2)

            target_r_samples_i = tf.stop_gradient(self.target_dist.sample(tf.shape(r_i)))
            target_r_samples_j = tf.stop_gradient(self.target_dist.sample(tf.shape(r_j)))

            qr1 = tf.clip_by_value(tf.sort(target_r_samples_i, axis=0), 0.0001, 0.9999)
            qr2 = tf.clip_by_value(tf.sort(target_r_samples_j, axis=0), 0.0001, 0.9999)
            r1 = tf.clip_by_value(tf.sort(r_i, axis=0), 0.0001, 0.9999)
            r2 = tf.clip_by_value(tf.sort(r_j, axis=0), 0.0001, 0.9999)

            r_loss = self.kl(r1, qr1) + self.kl(r2, qr2)

            loss_relevance = tf.reduce_mean(binary_crossentropy(y_true=y_true, y_pred=r_i))
            loss_propensity = tf.reduce_mean(binary_crossentropy(y_true=y_true, y_pred=p_i))

            reg_loss = 0.0001 * (tf.add_n(self.losses) + film_reg_loss_1 + film_reg_loss_2) + self.p_weight * p_loss + self.r_weight * r_loss

            loss = (
                self.lambda_1 * loss_pair +
                loss_click +
                self.lambda_2 * loss_relevance +
                self.lambda_3 * loss_propensity +
                reg_loss
            )

        gradients = tape.gradient(loss, self.trainable_weights)
        self.estimator_optimizer.apply_gradients(zip(gradients, self.trainable_weights))

if __name__ == "__main__":
    pass
