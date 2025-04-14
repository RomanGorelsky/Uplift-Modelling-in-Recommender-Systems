import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import coo_matrix
import pickle
import os

class CJBPR:
    def __init__(self, train_df, vali_df=None, test_df=None, 
                 hidden_dim=100, learning_rate=0.001, reg=2e-5, 
                 alpha=500000, beta=0.5, C=6, neg_samples=5, 
                 batch_size=1024, epochs=100, display_interval=1):
        """
        Initialize CJBPR model with data and parameters
        
        Args:
            train_df: DataFrame with columns ['userId', 'itemId', 'outcome']
            vali_df: Optional validation DataFrame (same columns)
            test_df: Optional test DataFrame (same columns)
            hidden_dim: Dimension of latent factors
            learning_rate: Learning rate for optimization
            reg: Regularization parameter
            alpha: Exposure regularization parameter
            beta: Residual regularization parameter
            C: Number of components
            neg_samples: Negative sampling rate
            batch_size: Batch size for training
            epochs: Number of training epochs
            display_interval: Interval for displaying training progress
        """
        # Store parameters
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.reg = reg
        self.alpha = alpha
        self.beta = beta
        self.C = C
        self.neg = neg_samples
        self.batch_size = batch_size
        self.epochs = epochs
        self.display = display_interval
        
        # Preprocess data
        self._preprocess_data(train_df, vali_df, test_df)
        
        # Initialize TensorFlow session and build model
        self.sess = tf.Session()
        self._build_model()
        
    def _preprocess_data(self, train_df, vali_df, test_df):
        """Preprocess all input dataframes"""
        # Convert ratings to binary (>=4 is positive)
        train_df = train_df[train_df['outcome'] >= 1].copy()
        train_df = train_df.drop(columns=['outcome'])
        
        if vali_df is not None:
            vali_df = vali_df.copy()
            vali_df['outcome'] = (vali_df['outcome'] >= 1).astype(int)
        
        if test_df is not None:
            test_df = test_df.copy()
            test_df['outcome'] = (test_df['outcome'] >= 1).astype(int)
        
        # Get unique users and items
        all_users = set(train_df['userId'].unique())
        all_items = set(train_df['itemId'].unique())
        
        # Filter test users to only those with at least one positive in test and two in train
        if test_df is not None:
            test_users = []
            for u in test_df['userId'].unique():
                pos_in_test = np.sum(test_df[test_df['userId'] == u]['outcome'])
                pos_in_train = len(train_df[train_df['userId'] == u])
                if pos_in_test >= 1 and pos_in_train >= 2:
                    test_users.append(u)
            test_df = test_df[test_df['userId'].isin(test_users)]
        
        # Reindex users and items
        self.user_map = {u: i for i, u in enumerate(sorted(all_users))}
        self.item_map = {i: idx for idx, i in enumerate(sorted(all_items))}
        
        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map)
        
        # Apply mappings to all dataframes
        self.train_df = train_df.copy()
        self.train_df['userId'] = self.train_df['userId'].map(self.user_map)
        self.train_df['itemId'] = self.train_df['itemId'].map(self.item_map)
        
        if vali_df is not None:
            self.vali_df = vali_df.copy()
            self.vali_df['userId'] = self.vali_df['userId'].map(self.user_map)
            self.vali_df['itemId'] = self.vali_df['itemId'].map(self.item_map)
        else:
            self.vali_df = None
            
        if test_df is not None:
            self.test_df = test_df.copy()
            self.test_df['userId'] = self.test_df['userId'].map(self.user_map)
            self.test_df['itemId'] = self.test_df['itemId'].map(self.item_map)
        else:
            self.test_df = None
        
        # Split train into C components
        self.df_list = []
        len_train = len(self.train_df)
        df_len = int(len_train * 1. / self.C)
        left_idx = range(len_train)
        
        for i in range(self.C - 1):
            idx = np.random.choice(left_idx, int(df_len), replace=False).tolist()
            self.df_list.append(self.train_df.iloc[idx].copy())
            left_idx = list(set(left_idx) - set(idx))
        self.df_list.append(self.train_df.iloc[left_idx].copy())
        
        # Compute item popularity
        self.item_pop = np.array(self.train_df['itemId'].value_counts(normalize=True))
        self.item_pop = self.item_pop.reshape((-1, 1))
        
        # Create user-item interaction lists
        self.train_like = [[] for _ in range(self.num_users)]
        for u, i in zip(self.train_df['userId'], self.train_df['itemId']):
            self.train_like[u].append(i)
        
        if self.vali_df is not None:
            self.vali_like = [[] for _ in range(self.num_users)]
            for u, i, r in zip(self.vali_df['userId'], self.vali_df['itemId'], self.vali_df['outcome']):
                if r == 1:
                    self.vali_like[u].append(i)
        else:
            self.vali_like = None
            
        if self.test_df is not None:
            self.test_users = list(self.test_df['userId'].unique())
            self.test_interact = [[] for _ in range(len(self.test_users))]
            self.test_like = [[] for _ in range(len(self.test_users))]
            
            for idx, u in enumerate(self.test_users):
                user_df = self.test_df[self.test_df['userId'] == u]
                self.test_interact[idx] = user_df['itemId'].values
                self.test_like[idx] = user_df[user_df['outcome'] == 1]['itemId'].values
        else:
            self.test_users = None
            self.test_interact = None
            self.test_like = None
    
    def _build_model(self):
        """Build the CJBPR TensorFlow model"""
        # Input placeholders
        self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
        self.item_input_pos = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_pos")
        self.item_input_neg = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_neg")
        self.pop_input_pos = tf.placeholder(tf.float32, shape=[None, 1], name="pop_input_pos")
        self.pop_input_neg = tf.placeholder(tf.float32, shape=[None, 1], name="pop_input_neg")
        self.rel_input = tf.placeholder(tf.float32, shape=[None, 1], name="rel_input")
        self.exp_input = tf.placeholder(tf.float32, shape=[None, 1], name="exp_input")
        
        # Initialize model parameters for each component
        self._init_parameters()
        
        # Build loss functions and optimizers
        self._build_loss_and_optimizer()
    
    def _init_parameters(self):
        """Initialize model parameters"""
        self.P_list = []
        self.Q_list = []
        self.c_list = []
        self.d_list = []
        self.a_list = []
        self.b_list = []
        self.e_list = []
        self.f_list = []
        
        for m in range(self.C):
            with tf.variable_scope(f'Component_{m}'):
                # Relevance parameters
                P = tf.get_variable(f'P_{m}', 
                                  shape=[self.num_users, self.hidden_dim],
                                  initializer=tf.truncated_normal_initializer(stddev=0.03))
                Q = tf.get_variable(f'Q_{m}',
                                  shape=[self.num_items, self.hidden_dim],
                                  initializer=tf.truncated_normal_initializer(stddev=0.03))
                
                # Exposure parameters
                c = tf.get_variable(f'c_{m}', shape=[self.hidden_dim, 1],
                                  initializer=tf.truncated_normal_initializer(stddev=0.03))
                d = tf.get_variable(f'd_{m}', shape=[1, 1],
                                  initializer=tf.truncated_normal_initializer(stddev=0.03))
                a = tf.get_variable(f'a_{m}', shape=[self.hidden_dim, 1],
                                  initializer=tf.truncated_normal_initializer(stddev=0.03))
                b = tf.get_variable(f'b_{m}', shape=[1, 1],
                                  initializer=tf.truncated_normal_initializer(stddev=0.03))
                e = tf.get_variable(f'e_{m}', shape=[self.hidden_dim, 1],
                                  initializer=tf.truncated_normal_initializer(stddev=0.03))
                f = tf.get_variable(f'f_{m}', shape=[1, 1],
                                  initializer=tf.truncated_normal_initializer(stddev=0.03))
                
                self.P_list.append(P)
                self.Q_list.append(Q)
                self.c_list.append(c)
                self.d_list.append(d)
                self.a_list.append(a)
                self.b_list.append(b)
                self.e_list.append(e)
                self.f_list.append(f)
    
    def _build_loss_and_optimizer(self):
        """Build loss functions and optimizers"""
        self.rel_cost_list = []
        self.exp_cost_list = []
        self.rel_reg_cost_list = []
        self.exp_reg_cost_list = []
        self.rel_optimizer_list = []
        self.exp_optimizer_list = []
        
        for m in range(self.C):
            P = self.P_list[m]
            Q = self.Q_list[m]
            c = self.c_list[m]
            d = self.d_list[m]
            a = self.a_list[m]
            b = self.b_list[m]
            e = self.e_list[m]
            f = self.f_list[m]
            
            # Get embeddings
            p = tf.nn.embedding_lookup(P, tf.reshape(self.user_input, [-1]))
            q_pos = tf.nn.embedding_lookup(Q, tf.reshape(self.item_input_pos, [-1]))
            q_neg = tf.nn.embedding_lookup(Q, tf.reshape(self.item_input_neg, [-1]))
            
            # Relevance prediction
            rel_pos = tf.reduce_sum(p * q_pos, 1, keepdims=True)
            rel_neg = tf.reduce_sum(p * q_neg, 1, keepdims=True)
            
            # Exposure prediction
            w_pos = tf.nn.sigmoid(tf.matmul(q_pos, a) + b)
            exp_pos = tf.pow(w_pos * tf.nn.sigmoid(tf.matmul(q_pos, c) + d) + 
                           (1 - w_pos) * self.pop_input_pos,
                           tf.nn.sigmoid(tf.matmul(q_pos, e) + f))
            exp_pos = tf.clip_by_value(exp_pos, 0.01, 0.99)
            
            w_neg = tf.nn.sigmoid(tf.matmul(q_neg, a) + b)
            exp_neg = tf.pow(w_neg * tf.nn.sigmoid(tf.matmul(q_neg, c) + d) + 
                     (1 - w_neg) * self.pop_input_neg,
                     tf.nn.sigmoid(tf.matmul(q_neg, e) + f))
            exp_neg = tf.clip_by_value(exp_neg, 0.01, 0.99)
            
            # Loss functions
            rel_cost = -tf.reduce_mean(tf.log(tf.nn.sigmoid(rel_pos - rel_neg)) / self.exp_input)
            exp_cost = -tf.reduce_mean(tf.log(exp_pos) / self.rel_input) - \
                       tf.reduce_mean(tf.log(1 - exp_neg))
            
            # Regularization
            rel_reg = self.reg * 0.5 * (tf.reduce_sum(tf.square(P)) + tf.reduce_sum(tf.square(Q)))
            exp_reg = self.alpha * self.reg * 0.5 * (
                tf.reduce_sum(tf.square(c)) + tf.reduce_sum(tf.square(d)) +
                tf.reduce_sum(tf.square(a)) + tf.reduce_sum(tf.square(b)) +
                tf.reduce_sum(tf.square(e)) + tf.reduce_sum(tf.square(f)))
            
            # Total loss
            rel_total = rel_cost + rel_reg
            exp_total = exp_cost + exp_reg
            
            # Optimizers
            rel_optimizer = tf.train.AdamOptimizer(self.lr).minimize(rel_total)
            exp_optimizer = tf.train.AdamOptimizer(self.lr).minimize(exp_total)
            
            self.rel_cost_list.append(rel_cost)
            self.exp_cost_list.append(exp_cost)
            self.rel_reg_cost_list.append(rel_reg)
            self.exp_reg_cost_list.append(exp_reg)
            self.rel_optimizer_list.append(rel_optimizer)
            self.exp_optimizer_list.append(exp_optimizer)
    
    def fit(self):
        """Train the model"""
        self.sess.run(tf.global_variables_initializer())
        
        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)
            
            if epoch % self.display == 0:
                if self.vali_like is not None:
                    self._validate(epoch)
                if self.test_users is not None:
                    self._test(epoch)
    
    def _train_epoch(self, epoch):
        """Train for one epoch"""
        # Train each component
        for m in range(self.C):
            df = self.df_list[m]
            user_list, item_pos_list, item_neg_list = self._negative_sampling(df)
            
            pop_pos = self.item_pop[item_pos_list.reshape(-1)].reshape((-1, 1))
            pop_neg = self.item_pop[item_neg_list.reshape(-1)].reshape((-1, 1))
            
            num_batches = int(len(user_list) / self.batch_size) + 1
            idx = np.random.permutation(len(user_list))
            
            for i in range(num_batches):
                batch_idx = idx[i*self.batch_size:(i+1)*self.batch_size] if i < num_batches-1 else idx[i*self.batch_size:]
                
                feed_dict = {
                    self.user_input: user_list[batch_idx],
                    self.item_input_pos: item_pos_list[batch_idx],
                    self.item_input_neg: item_neg_list[batch_idx],
                    self.pop_input_pos: pop_pos[batch_idx],
                    self.pop_input_neg: pop_neg[batch_idx],
                    self.rel_input: np.ones((len(batch_idx), 1)),  # Placeholder
                    self.exp_input: np.ones((len(batch_idx), 1))   # Placeholder
                }
                
                # Update relevance and exposure for this component
                _, rel_cost, _, exp_cost = self.sess.run(
                    [self.rel_optimizer_list[m], self.rel_cost_list[m],
                     self.exp_optimizer_list[m], self.exp_cost_list[m]],
                    feed_dict=feed_dict)
                
            if epoch % self.display == 0:
                print(f"Epoch {epoch}, Component {m}: Rel Cost={rel_cost:.4f}, Exp Cost={exp_cost:.4f}")
    
    def _negative_sampling(self, df):
        """Generate negative samples for BPR loss"""
        pos_users = df['userId'].values.reshape((-1, 1))
        pos_items = df['itemId'].values.reshape((-1, 1))
        
        # Repeat each positive sample neg times
        users = np.tile(pos_users, (self.neg, 1))
        pos = np.tile(pos_items, (self.neg, 1))
        
        # Generate negative samples
        neg = np.random.randint(0, self.num_items, size=(len(pos), 1))
        
        # Remove cases where neg == pos
        mask = (neg != pos).reshape(-1)
        users = users[mask]
        pos = pos[mask]
        neg = neg[mask]
        
        return users, pos, neg
    
    def _validate(self, epoch):
        """Validate model performance"""
        # Generate recommendations for validation set
        R = np.zeros((self.num_users, self.num_items))
        
        for m in range(self.C):
            P, Q = self.sess.run([self.P_list[m], self.Q_list[m]])
            R += np.matmul(P, Q.T)
        
        R /= self.C
        
        # Evaluate on validation set
        recall = self._evaluate_recall(R, self.train_like, self.vali_like)
        print(f"Validation @ Epoch {epoch}: Recall={recall:.4f}")
    
    def _test(self, epoch):
        """Test model performance"""
        # Generate recommendations
        R = np.zeros((self.num_users, self.num_items))
        
        for m in range(self.C):
            P, Q = self.sess.run([self.P_list[m], self.Q_list[m]])
            R += np.matmul(P, Q.T)
        
        R /= self.C
        
        # Evaluate on test set
        avg_recall = 0
        for i, u in enumerate(self.test_users):
            test_items = self.test_interact[i]
            test_likes = self.test_like[i]
            
            if len(test_likes) == 0:
                continue
                
            scores = R[u, test_items]
            ranked = np.argsort(-scores)
            
            # Compute recall@k
            hits = 0
            for k, idx in enumerate(ranked[:10]):  # recall@10
                if test_items[idx] in test_likes:
                    hits += 1
            recall = hits / len(test_likes)
            avg_recall += recall
        
        avg_recall /= len(self.test_users)
        print(f"Test @ Epoch {epoch}: Avg Recall@10={avg_recall:.4f}")
    
    def _evaluate_recall(self, R, train_like, test_like):
        """Compute average recall@10"""
        total_recall = 0
        count = 0
        
        for u in range(self.num_users):
            if len(test_like[u]) == 0:
                continue
                
            # Get top items not in training
            scores = R[u].copy()
            scores[train_like[u]] = -np.inf
            top_items = np.argsort(-scores)[:10]
            
            # Compute recall
            hits = len(set(top_items) & set(test_like[u]))
            recall = hits / len(test_like[u])
            total_recall += recall
            count += 1
        
        return total_recall / count if count > 0 else 0
    
    def predict(self, users, items=None, k=10):
        """Generate recommendations for users
        
        Args:
            users: List of user IDs (original IDs, not mapped)
            items: Optional list of candidate items (original IDs)
            k: Number of recommendations to return
            
        Returns:
            Dictionary mapping user IDs to recommended item IDs
        """
        # Convert user and item IDs to internal indices
        user_indices = [self.user_map[u] for u in users if u in self.user_map]
        
        if items is not None:
            item_indices = [self.item_map[i] for i in items if i in self.item_map]
        else:
            item_indices = None
        
        # Generate recommendations
        R = np.zeros((self.num_users, self.num_items))
        for m in range(self.C):
            P, Q = self.sess.run([self.P_list[m], self.Q_list[m]])
            R += np.matmul(P, Q.T)
        R /= self.C
        
        recommendations = {}
        for u in users:
            if u not in self.user_map:
                recommendations[u] = []
                continue
                
            u_idx = self.user_map[u]
            scores = R[u_idx].copy()
            
            # Remove items already interacted with
            scores[self.train_like[u_idx]] = -np.inf
            
            # Filter to candidate items if provided
            if item_indices is not None:
                mask = np.zeros(self.num_items, dtype=bool)
                mask[item_indices] = True
                scores[~mask] = -np.inf
            
            # Get top k items
            top_items = np.argsort(-scores)[:k]
            recommendations[u] = [list(self.item_map.keys())[list(self.item_map.values()).index(i)] 
                                for i in top_items if i in self.item_map.values()]
        
        return recommendations
    
    def save(self, path):
        """Save model to directory"""
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save parameters
        params = {
            'hidden_dim': self.hidden_dim,
            'lr': self.lr,
            'reg': self.reg,
            'alpha': self.alpha,
            'beta': self.beta,
            'C': self.C,
            'neg': self.neg,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'display': self.display,
            'num_users': self.num_users,
            'num_items': self.num_items,
            'user_map': self.user_map,
            'item_map': self.item_map,
            'item_pop': self.item_pop
        }
        
        with open(os.path.join(path, 'params.pkl'), 'wb') as f:
            pickle.dump(params, f)
        
        # Save TensorFlow model
        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(path, 'model.ckpt'))
    
    @classmethod
    def load(cls, path):
        """Load model from directory"""
        # Load parameters
        with open(os.path.join(path, 'params.pkl'), 'rb') as f:
            params = pickle.load(f)
        
        # Create dummy instance
        dummy_df = pd.DataFrame({'userId': [0], 'itemId': [0], 'outcome ': [1]})
        model = cls(dummy_df, hidden_dim=params['hidden_dim'], 
                   learning_rate=params['lr'], reg=params['reg'],
                   alpha=params['alpha'], beta=params['beta'], 
                   C=params['C'], neg_samples=params['neg'],
                   batch_size=params['batch_size'], epochs=params['epochs'],
                   display_interval=params['display'])
        
        # Restore actual parameters
        model.num_users = params['num_users']
        model.num_items = params['num_items']
        model.user_map = params['user_map']
        model.item_map = params['item_map']
        model.item_pop = params['item_pop']
        
        # Restore TensorFlow model
        saver = tf.train.Saver()
        saver.restore(model.sess, os.path.join(path, 'model.ckpt'))
        
        return model