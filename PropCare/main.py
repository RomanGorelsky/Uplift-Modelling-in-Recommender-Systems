import argparse
from train import prepare_data, train_propensity
from train import plotpath, Causal_Model
from baselines import DLMF, PopularBase, MF, CausalNeighborBase, DLMFMod
import numpy as np
# from CJBPR import CJBPR
import tensorflow as tf
from evaluator import Evaluator
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import gaussian_kde
import pickle
import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--dimension", default=128, type=int, help="number of features per user/item.")
parser.add_argument("--estimator_layer_units",
                    default=[64, 32, 16, 8],
                    type=list,
                    help="number of nodes each layer for MLP layers in Propensity and Relevance estimators")
parser.add_argument("--embedding_layer_units",
                    default=[256, 128, 64],
                    type=list,
                    help="number of nodes each layer for shared embedding layer.")
parser.add_argument("--click_layer_units",
                    default=[64, 32, 16, 8],
                    type=list,
                    help="number of nodes each layer for MLP layers in Click estimators")
parser.add_argument("--epoch", default=25, type=int,
                    help="Number of epochs in the training")
parser.add_argument("--prop_train", default=False, type=bool,
                    help="Start the training of PropCare")
parser.add_argument("--rec_train", default=False, type=bool,
                    help="Start the training of Recommender")
parser.add_argument("--lambda_1", default=10.0, type=float,
                    help="weight for popularity loss.")
parser.add_argument("--lambda_2", default=0.1, type=float,
                    help="weight for relavance loss.")
parser.add_argument("--lambda_3", default=0.1, type=float,
                    help="weight for propensity2 loss.")
parser.add_argument("--dataset", default='d', type=str,
                    help="the dataset used")
parser.add_argument("--batch_size", default=5096, type=int,
                    help="the batch size")
parser.add_argument("--repeat", default=1, type=int,
                    help="how many time to run the model")
parser.add_argument("--add", default='default', type=str,
                    help="additional information")
parser.add_argument("--p_weight", default=0.4, type=float,
                    help="weight for p_loss")
parser.add_argument("--r_weight", default=0.4, type=float,
                    help="weight for r_loss")
parser.add_argument("--saved_DLMF", default='n', type=str,
                    help="use saved weights of DLMF")
parser.add_argument("--to_prob", default=True, type=bool,
                    help="normalize as probability")
flag = parser.parse_args()


def main(flag=flag):

    flag.add = flag.add + '/' + flag.dataset[-1] + '/'

    cp10list_pred = []
    cp100list_pred = []
    cdcglist_pred = []

    cp10list_rel = []
    cp100list_rel = []
    cdcglist_rel = []

    cp10list_pop = []
    cp100list_pop = []
    cdcglist_pop = []

    cp10list_pers_pop = []
    cp100list_pers_pop = []
    cdcglist_pers_pop = []

    ndcglist_rel = []
    ndcglist_pred = []
    ndcglist_pop = []
    ndcglist_pers_pop = []

    recalllist_rel = []
    recalllist_pred = []
    recalllist_pop = []
    recalllist_pers_pop = []

    precisionlist_rel = []
    precisionlist_pred = []
    precisionlist_pop = []
    precisionlist_pers_pop = []

    random_seed = int(240)
    
    for epoch in range(flag.repeat):
        train_df, vali_df, test_df, num_users, num_items, num_times, popular = prepare_data(flag)
        random_seed += 1
        tf.random.set_seed(
            random_seed
        )
        model = train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular)
        print(test_df['personal_popular'].value_counts())

        train_user = tf.convert_to_tensor(train_df["idx_user"].to_numpy(), dtype=tf.int32)
        train_item = tf.convert_to_tensor(train_df["idx_item"].to_numpy(), dtype=tf.int64)
        train_data = tf.data.Dataset.from_tensor_slices((train_user, train_item))

        val_user = tf.convert_to_tensor(vali_df["idx_user"].to_numpy(), dtype=tf.int32)
        val_item = tf.convert_to_tensor(vali_df["idx_item"].to_numpy(), dtype=tf.int64)
        val_data = tf.data.Dataset.from_tensor_slices((val_user, val_item))

        test_user = tf.convert_to_tensor(test_df["idx_user"].to_numpy(), dtype=tf.int32)
        test_item = tf.convert_to_tensor(test_df["idx_item"].to_numpy(), dtype=tf.int64)
        test_data = tf.data.Dataset.from_tensor_slices((test_user, test_item))

        opt_scale = 0.25
        opt_add = 0.5
        # opt_epsilon = 0.7
        # opt_c = 0.8

        # scales = np.linspace(0.3, 0.6, 9)
        # adds = np.linspace(0, 0.2, 9)
        # epsilons = np.linspace(0.5, 0.9, 9)
        # cs = np.linspace(0.1, 0.9, 100)

        p_pred = None

        for u, i in train_data.batch(5000):
            _, p_batch, _, _ = model((u, i), training=False)
            if p_pred is None:
                p_pred = p_batch
            else:
                p_pred = tf.concat((p_pred, p_batch), axis=0)

        p_pred = p_pred.numpy()
        # p_pred_true = np.squeeze(train_df["propensity"].to_numpy())

        p_pred_t = opt_scale * ((p_pred - np.mean(p_pred))/ (np.std(p_pred)))
        p_pred_t = np.clip((p_pred_t + opt_add), 0.0, 1.0)

        # t_pred_t = np.where(p_pred_t >= opt_epsilon, 1.0, 0.0)
        # max_f = f1_score(train_df['treated'], t_pred_t)
        # p_pred_t = p_pred * opt_c
        # p_pred_t = np.clip(p_pred_t, 0.0001, 0.9999)
        # p_pred_t = np.squeeze(p_pred_t)
        # roc_max = roc_auc_score(train_df['treated'], p_pred_t)
        # print('Initial F1 score', max_f)
        # print('Initial ROC-AUC', roc_max)

        # for scale in scales:
        #     for add in adds:
        #         for epsilon in epsilons:
        #             p_pred_t = scale * ((p_pred - np.mean(p_pred))/ (np.std(p_pred)))
        #             p_pred_t = np.clip((p_pred_t + add), 0.0, 1.0)
        #             t_pred_t = np.where(p_pred_t >= epsilon, 1.0, 0.0)
        #             f_score = f1_score(train_df['treated'], t_pred_t)
        #             if f_score > max_f:
        #                 max_f = f_score
        #                 opt_scale = scale
        #                 opt_add = add
        #                 opt_epsilon = epsilon
        
        # for c in cs:
        #     p_pred_t = p_pred * c
        #     p_pred_t = np.clip(p_pred_t, 0.0001, 0.9999)
        #     p_pred_t = np.squeeze(p_pred_t)
        #     roc = roc_auc_score(train_df['treated'], p_pred_t)
        #     if roc > roc_max:
        #         roc_max = roc
        #         opt_c = c
        
        # print('Max F1 score: ', max_f)
        # print('Max ROC-AUC: ', roc_max)
        # print('Optimal scale: ', opt_scale)
        # print('Optimal add: ', opt_add)
        # print('Optimal epsilon: ', opt_epsilon)
        # print('Optimal c: ', opt_c)

        # break

        if flag.dataset[-1] == "d" or "p":
            flag.thres = 0.7

            if flag.dataset[-1] == "d":
                opt_c = 0.9
            else:
                opt_c = 0.8
        elif flag.dataset == "ml":
            flag.thres = 0.65

        t_pred = np.where(p_pred_t >= flag.thres, 1.0, 0.0)

        if flag.dataset == "d" or "p":
            p_pred = p_pred * opt_c
        if flag.dataset == "ml":
            p_pred = p_pred * 0.2
        
        train_df["propensity"] = np.clip(p_pred, 0.0001, 0.9999)
        train_df["treated"] = t_pred

        if flag.dataset[-1] == "d":
            cap = 0.03
            lr = 0.001
            rf = 0.01
            itr = 100e6
        if flag.dataset[-1] == "p":
            lr = 0.001
            cap = 0.5
            rf = 0.001
            itr = 100e6
        else:
            lr = 0.001
            cap = 0.3
            rf = 0.1
            itr = 100e6

        # recommender = DLMF(num_users, num_items, capping_T = cap, 
        #                 capping_C = cap, learn_rate = lr, reg_factor = rf)

        recommender = DLMFMod(num_users, num_items, capping_T = cap, 
                        capping_C = cap, learn_rate = lr, reg_factor = rf)
        
        if flag.rec_train:
            recommender.train(train_df, plotpath + flag.add, iter=itr)
        
        else:
            with open(plotpath + flag.add + "dlmf_weights.pkl", "rb") as f:
                saved_state = pickle.load(f)
            recommender.__dict__.update(saved_state)
            print("DLMF weights loaded successfully!")

        cp10_tmp_list_pred = []
        cp100_tmp_list_pred = []
        cdcg_tmp_list_pred = []

        cp10_tmp_list_rel = []
        cp100_tmp_list_rel = []
        cdcg_tmp_list_rel = []

        cp10_tmp_list_pop = []
        cp100_tmp_list_pop = []
        cdcg_tmp_list_pop = []

        cp10_tmp_list_pers_pop = []
        cp100_tmp_list_pers_pop = []
        cdcg_tmp_list_pers_pop = []
        
        ndcg_tmp_list_rel = []
        ndcg_tmp_list_pred = []
        ndcg_tmp_list_pop = []
        ndcg_tmp_list_pers_pop = []

        recall_tmp_list_rel = []
        recall_tmp_list_pred = []
        recall_tmp_list_pop = []
        recall_tmp_list_pers_pop = []

        precision_tmp_list_rel = []
        precision_tmp_list_pred = []
        precision_tmp_list_pop = []
        precision_tmp_list_pers_pop = []

        if flag.dataset[-1] == 'd' or 'p':
            for t in range(num_times):
                test_df_t = test_df[test_df["idx_time"] == t]
                user = tf.convert_to_tensor(test_df_t["idx_user"].to_numpy(), dtype=tf.int32)
                item = tf.convert_to_tensor(test_df_t["idx_item"].to_numpy(), dtype=tf.int64)
                test_t_data = tf.data.Dataset.from_tensor_slices((user, item))
                r_pred_test = None
                p_pred_test = None

                for u, i in test_t_data.batch(5000):
                    _, p_batch, r_batch, _ = model((u, i), training=False)
                    if r_pred_test is None:
                        r_pred_test = r_batch
                        p_pred_test = p_batch
                    else:
                        r_pred_test = tf.concat((r_pred_test, r_batch), axis=0)
                        p_pred_test = tf.concat((p_pred_test, p_batch), axis=0)

                p_pred_test = p_pred_test.numpy()
                r_pred_test = r_pred_test.numpy()
                p_pred_test_t = opt_scale * ((p_pred_test - np.mean(p_pred_test))/ (np.std(p_pred_test)))
                p_pred_test_t = np.clip((p_pred_test_t + opt_add), 0.0, 1.0)

                t_test_pred = np.where(p_pred_test_t >= flag.thres, 1.0, 0.0)

                p_pred_test = p_pred_test * opt_c
                r_pred_test = r_pred_test * opt_c

                test_df_t["propensity_estimate"] = np.clip(p_pred_test, 0.0001, 0.9999)
                test_df_t["relevance_estimate"] = np.clip(r_pred_test, 0.0001, 0.9999)

                outcome_estimate = test_df_t["propensity_estimate"] * test_df_t["relevance_estimate"]
                outcome_estimate = opt_scale * ((outcome_estimate - np.mean(outcome_estimate))/ (np.std(outcome_estimate)))
                outcome_estimate = np.clip((outcome_estimate + opt_add), 0.0, 1.0)
                test_df_t["outcome_estimate"] = np.where(outcome_estimate >= flag.thres, 1.0, 0.0)
                test_df_t["treated_estimate"] = t_test_pred
                causal_effect_estimate = \
                    test_df_t["outcome_estimate"] * \
                    (test_df_t["treated_estimate"] / test_df_t["propensity_estimate"] - \
                    (1 - test_df_t["treated_estimate"]) / (1 - test_df_t["propensity_estimate"]))
                test_df_t["causal_effect_estimate"] = np.clip(causal_effect_estimate, -1, 1)

                train_df = train_df[train_df.outcome>0]
                popularity = train_df["idx_item"].value_counts().reset_index()
                popularity.columns = ["idx_item", "popularity"]
                test_df_t = test_df_t.merge(popularity, on="idx_item", how="left")
                test_df_t['popularity'] = (test_df_t['popularity'] - np.min(test_df_t['popularity'])) \
                                            / (np.max(test_df_t['popularity']) - np.min(test_df_t['popularity']))
                test_df_t['frequency'] = test_df_t['personal_popular']
                test_df_t['personal_popular'] = test_df_t['personal_popular'] + test_df_t['popularity']

                test_df_t["pred"] = recommender.predict(test_df_t)

                evaluator = Evaluator()

                cp10_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrecS', 10))
                cp100_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrecS', 100))
                cdcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CDCGS', 100000))

                cp10_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 10))
                cp100_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 100))
                cdcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CDCGR', 100000))

                cp10_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 10))
                cp100_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 100))
                cdcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CDCGP', 100000))

                cp10_tmp_list_pers_pop.append(evaluator.evaluate(test_df_t, 'CPrecPP', 10))
                cp100_tmp_list_pers_pop.append(evaluator.evaluate(test_df_t, 'CPrecPP', 100))
                cdcg_tmp_list_pers_pop.append(evaluator.evaluate(test_df_t, 'CDCGPP', 100000))

                ndcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'NDCGR', 10))
                ndcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'NDCGS', 10))
                ndcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'NDCGP', 10))
                ndcg_tmp_list_pers_pop.append(evaluator.evaluate(test_df_t, 'NDCGPP', 10))

                recall_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'RecallR', 10))
                recall_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'RecallS', 10))
                recall_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'RecallP', 10))
                recall_tmp_list_pers_pop.append(evaluator.evaluate(test_df_t, 'RecallPP', 10))

                precision_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'PrecisionR', 10))
                precision_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'PrecisionS', 10))
                precision_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'PrecisionP', 10))
                precision_tmp_list_pers_pop.append(evaluator.evaluate(test_df_t, 'PrecisionPP', 10))

                kendall_score = evaluator.kendall_tau_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
                spearman_score = evaluator.spearman_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
                pos_diff = evaluator.avg_position_diff(test_df_t, 'idx_user', 'idx_item', 'pred', 'relevance_estimate')

                print(f"Kendall Tau: {kendall_score:.4f}")
                print(f"Spearman Rho: {spearman_score:.4f}")
                print(f"Average Rank Position Difference: {pos_diff:.4f}")

                if t + 1 == num_times:
                    evaluator.get_dataframes(test_df_t, plotpath + flag.add)
        else:
            for t in [0]:
                test_df_t = test_df[test_df["idx_time"] == t]
                user = tf.convert_to_tensor(test_df_t["idx_user"].to_numpy(), dtype=tf.int32)
                item = tf.convert_to_tensor(test_df_t["idx_item"].to_numpy(), dtype=tf.int64)
                test_t_data = tf.data.Dataset.from_tensor_slices((user, item))
                r_pred_test = None
                p_pred_test = None

                for u, i in test_t_data.batch(5000):
                    _, p_batch, r_batch, _ = model((u, i), training=False)
                    if r_pred_test is None:
                        r_pred_test = r_batch
                        p_pred_test = p_batch
                    else:
                        r_pred_test = tf.concat((r_pred_test, r_batch), axis=0)
                        p_pred_test = tf.concat((p_pred_test, p_batch), axis=0)

                p_pred_test = p_pred_test.numpy()
                r_pred_test = r_pred_test.numpy()
                p_pred_test_t = opt_scale * ((p_pred_test - np.mean(p_pred_test))/ (np.std(p_pred_test)))
                p_pred_test_t = np.clip((p_pred_test_t + opt_add), 0.0, 1.0)

                t_test_pred = np.where(p_pred_test_t >= flag.thres, 1.0, 0.0)
                p_pred_test = p_pred_test * 0.2
                r_pred_test = r_pred_test * 0.2
                test_df_t["propensity_estimate"] = np.clip(p_pred_test, 0.0001, 0.9999)
                test_df_t["relevance_estimate"] = np.clip(r_pred_test, 0.0001, 0.9999)
                outcome_estimate = test_df_t["propensity_estimate"] * test_df_t["relevance_estimate"]
                outcome_estimate = opt_scale * ((outcome_estimate - np.mean(outcome_estimate))/ (np.std(outcome_estimate)))
                outcome_estimate = np.clip((outcome_estimate + opt_add), 0.0, 1.0)
                test_df_t["outcome_estimate"] = np.where(outcome_estimate >= flag.thres, 1.0, 0.0)
                test_df_t["treated_estimate"] = t_test_pred
                causal_effect_estimate = \
                    test_df_t["outcome_estimate"] * \
                    (test_df_t["treated_estimate"] / test_df_t["propensity_estimate"] - \
                    (1 - test_df_t["treated_estimate"]) / (1 - test_df_t["propensity_estimate"]))
                test_df_t["causal_effect_estimate"] = np.clip(causal_effect_estimate, -1, 1)

                train_df = train_df[train_df.outcome>0]
                popularity = train_df["idx_item"].value_counts().reset_index()
                popularity.columns = ["idx_item", "popularity"]
                test_df_t = test_df_t.merge(popularity, on="idx_item", how="left")
                test_df_t["pred"] = recommender.predict(test_df_t)
                evaluator = Evaluator()

                cp10_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrecS', 10))
                cp100_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrecS', 100))
                cdcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CDCGS', 100000))

                cp10_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 10))
                cp100_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 100))
                cdcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CDCGR', 100000))

                cp10_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 10))
                cp100_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 100))
                cdcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CDCGP', 100000))

                ndcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'NDCGR', 10))
                ndcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'NDCGS', 10))
                ndcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'NDCGP', 10))

                recall_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'RecallR', 10))
                recall_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'RecallS', 10))
                recall_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'RecallP', 10))

                precision_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'PrecisionR', 10))
                precision_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'PrecisionS', 10))
                precision_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'PrecisionP', 10))

                kendall_score = evaluator.kendall_tau_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
                spearman_score = evaluator.spearman_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
                pos_diff = evaluator.avg_position_diff(test_df_t, 'idx_user', 'idx_item', 'pred', 'relevance_estimate')

                print(f"Kendall Tau: {kendall_score:.4f}")
                print(f"Spearman Rho: {spearman_score:.4f}")
                print(f"Average Rank Position Difference: {pos_diff:.4f}")

                evaluator.get_dataframes(test_df_t, plotpath + flag.add)

        cp10_pred = np.mean(cp10_tmp_list_pred)
        cp100_pred = np.mean(cp100_tmp_list_pred)
        cdcg_pred = np.mean(cdcg_tmp_list_pred)

        cp10_rel = np.mean(cp10_tmp_list_rel)
        cp100_rel = np.mean(cp100_tmp_list_rel)
        cdcg_rel = np.mean(cdcg_tmp_list_rel)

        cp10_pop = np.mean(cp10_tmp_list_pop)
        cp100_pop = np.mean(cp100_tmp_list_pop)
        cdcg_pop = np.mean(cdcg_tmp_list_pop)

        cp10_pers_pop = np.mean(cp10_tmp_list_pers_pop)
        cp100_pers_pop = np.mean(cp100_tmp_list_pers_pop)
        cdcg_pers_pop = np.mean(cdcg_tmp_list_pers_pop)

        ndcg_rel = np.mean(ndcg_tmp_list_rel)
        ndcg_pred = np.mean(ndcg_tmp_list_pred)
        ndcg_pop = np.mean(ndcg_tmp_list_pop)
        ndcg_pers_pop = np.mean(ndcg_tmp_list_pers_pop)

        recall_rel = np.mean(recall_tmp_list_rel)
        recall_pred = np.mean(recall_tmp_list_pred)
        recall_pop = np.mean(recall_tmp_list_pop)
        recall_pers_pop = np.mean(recall_tmp_list_pers_pop)

        precision_rel = np.mean(precision_tmp_list_rel)
        precision_pred = np.mean(precision_tmp_list_pred)
        precision_pop = np.mean(precision_tmp_list_pop)
        precision_pers_pop = np.mean(precision_tmp_list_pers_pop)

        cp10list_pred.append(cp10_pred)
        cp100list_pred.append(cp100_pred)
        cdcglist_pred.append(cdcg_pred)

        cp10list_rel.append(cp10_rel)
        cp100list_rel.append(cp100_rel)
        cdcglist_rel.append(cdcg_rel)

        cp10list_pop.append(cp10_pop)
        cp100list_pop.append(cp100_pop)
        cdcglist_pop.append(cdcg_pop)

        cp10list_pers_pop.append(cp10_pers_pop)
        cp100list_pers_pop.append(cp100_pers_pop)
        cdcglist_pers_pop.append(cdcg_pers_pop)

        ndcglist_rel.append(ndcg_rel)
        ndcglist_pred.append(ndcg_pred)
        ndcglist_pop.append(ndcg_pop)
        ndcglist_pers_pop.append(ndcg_pers_pop)

        recalllist_rel.append(recall_rel)
        recalllist_pred.append(recall_pred)
        recalllist_pop.append(recall_pop)
        recalllist_pers_pop.append(recall_pers_pop)

        precisionlist_rel.append(precision_rel)
        precisionlist_pred.append(precision_pred)
        precisionlist_pop.append(precision_pop)
        precisionlist_pers_pop.append(precision_pers_pop)       

    with open(plotpath + "/result_" + flag.dataset +".txt", "a+") as f:
        print("CP10S:", np.mean(cp10list_pred), np.std(cp10list_pred), file=f)
        print("CP10R:", np.mean(cp10list_rel), np.std(cp10list_rel), file=f)
        print("CP10P:", np.mean(cp10list_pop), np.std(cp10list_pop), file=f)
        print("CP10PP:", np.mean(cp10list_pers_pop), np.std(cp10list_pers_pop), file=f)

        print("CP100S:", np.mean(cp100list_pred), np.std(cp100list_pred), file=f)
        print("CP100R:", np.mean(cp100list_rel), np.std(cp100list_rel), file=f)
        print("CP100P:", np.mean(cp100list_pop), np.std(cp100list_pop), file=f)
        print("CP100PP:", np.mean(cp100list_pers_pop), np.std(cp100list_pers_pop), file=f)
        
        print("CDCGS:", np.mean(cdcglist_pred), np.std(cdcglist_pred), file=f)
        print("CDCGR:", np.mean(cdcglist_rel), np.std(cdcglist_rel), file=f)
        print("CDCGP:", np.mean(cdcglist_pop), np.std(cdcglist_pop), file=f)
        print("CDCGPP:", np.mean(cdcglist_pers_pop), np.std(cdcglist_pers_pop), file=f)

        print("NDCG10R:", np.mean(ndcglist_rel), np.std(ndcglist_rel), file=f)
        print("NDCG10S:", np.mean(ndcglist_pred), np.std(ndcglist_pred), file=f)
        print("NDCG10P:", np.mean(ndcglist_pop), np.std(ndcglist_pop), file=f)
        print("NDCG10PP:", np.mean(ndcglist_pers_pop), np.std(ndcglist_pers_pop), file=f)

        print("Recall10R:", np.mean(recalllist_rel), np.std(recalllist_rel), file=f)
        print("Recall10S:", np.mean(recalllist_pred), np.std(recalllist_pred), file=f)
        print("Recall10P:", np.mean(recalllist_pop), np.std(recalllist_pop), file=f)
        print("Recall10PP:", np.mean(recalllist_pers_pop), np.std(recalllist_pers_pop), file=f)

        print("Precision10R:", np.mean(precisionlist_rel), np.std(precisionlist_rel), file=f)
        print("Precision10S:", np.mean(precisionlist_pred), np.std(precisionlist_pred), file=f)
        print("Precision10P:", np.mean(precisionlist_pop), np.std(precisionlist_pop), file=f) 
        print("Precision10PP:", np.mean(precisionlist_pers_pop), np.std(precisionlist_pers_pop), file=f) 
        print("--------------------------------", file=f)    
            
if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
    main(flag)
