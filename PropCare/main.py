import argparse
from train import prepare_data, train_propensity
from train import plotpath, Causal_Model
from baselines import DLMF
import numpy as np
import tensorflow as tf
from evaluator import Evaluator
import pickle
import os

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
parser.add_argument("--epoch", default=50, type=int,
                    help="Number of epochs in the training")
parser.add_argument("--lambda_1", default=10.0, type=float,
                    help="weight for popularity loss.")
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
parser.add_argument("--saved_DLMF", default='n', type=str,
                    help="use saved weights of DLMF")
flag = parser.parse_args()


def main(flag=flag):
    cp10list = []
    cp100list = []
    cdcglist = []

    ndcglist_rel = []
    ndcglist_pred = []
    ndcglist_pop = []

    recalllist_rel = []
    recalllist_pred = []
    recalllist_pop = []

    precisionlist_rel = []
    precisionlist_pred = []
    precisionlist_pop = []

    random_seed = int(233)
    for epoch in range(flag.repeat):
        train_df, vali_df, test_df, num_users, num_items, num_times, popular = prepare_data(flag)
        random_seed += 1
        tf.random.set_seed(
            random_seed
        )
        model = train_propensity(train_df, vali_df, test_df, flag, num_users, num_items, num_times, popular)
        train_user = tf.convert_to_tensor(train_df["idx_user"].to_numpy(), dtype=tf.int32)
        train_item = tf.convert_to_tensor(train_df["idx_item"].to_numpy(), dtype=tf.int64)
        train_data = tf.data.Dataset.from_tensor_slices((train_user, train_item))

        test_user = tf.convert_to_tensor(test_df["idx_user"].to_numpy(), dtype=tf.int32)
        test_item = tf.convert_to_tensor(test_df["idx_item"].to_numpy(), dtype=tf.int64)
        test_data = tf.data.Dataset.from_tensor_slices((test_user, test_item))
        p_pred = None

        for u, i in train_data.batch(5000):
            _, p_batch, _, _ = model((u, i), training=False)
            if p_pred is None:
                p_pred = p_batch
            else:
                p_pred = tf.concat((p_pred, p_batch), axis=0)

        p_pred = p_pred.numpy()
        p_pred_t = 0.25 * ((p_pred - np.mean(p_pred))/ (np.std(p_pred)))
        p_pred_t = np.clip((p_pred_t + 0.5), 0.0, 1.0)

        if flag.dataset == "d" or "p":
            flag.thres = 0.70
        elif flag.dataset == "ml":
            flag.thres = 0.65

        t_pred = np.where(p_pred_t >= flag.thres, 1.0, 0.0)
        if flag.dataset == "d" or "p":
            p_pred = p_pred * 0.8
        if flag.dataset == "ml":
            p_pred = p_pred * 0.2

        train_df["propensity"] = np.clip(p_pred, 0.0001, 0.9999)
        train_df["treated"] = t_pred

        if flag.dataset == "d":
            cap = 0.03
            lr = 0.001
            rf = 0.01
            itr = 100e6
        if flag.dataset == "p":
            lr = 0.001
            cap = 0.5
            rf = 0.001
            itr = 1e6
        if flag.dataset == "ml":
            lr = 0.001
            cap = 0.3
            rf = 0.1
            itr = 100e6

        with open("dlmf_weights.pkl", "rb") as f:
            saved_state = pickle.load(f)

        recommender = DLMF(num_users, num_items, capping_T = cap, 
                           capping_C = cap, learn_rate = lr, reg_factor = rf)

        recommender.__dict__.update(saved_state)
        print("DLMF weights loaded successfully!")

        # recommender = DLMF(num_users, num_items, capping_T=cap, capping_C=cap, learn_rate=lr, reg_factor=rf)
        # recommender.train(train_df, iter=itr)
        cp10_tmp_list = []
        cp100_tmp_list = []
        cdcg_tmp_list = []
        
        ndcg_tmp_list_rel = []
        ndcg_tmp_list_pred = []
        ndcg_tmp_list_pop = []

        recall_tmp_list_rel = []
        recall_tmp_list_pred = []
        recall_tmp_list_pop = []

        precision_tmp_list_rel = []
        precision_tmp_list_pred = []
        precision_tmp_list_pop = []

        if flag.dataset == 'd' or 'p':
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
                p_pred_test_t = 0.25 * ((p_pred_test - np.mean(p_pred_test))/ (np.std(p_pred_test)))
                p_pred_test_t = np.clip((p_pred_test_t + 0.5), 0.0, 1.0)

                t_test_pred = np.where(p_pred_test_t >= 0.7, 1.0, 0.0)
                p_pred_test = p_pred_test * 0.8
                r_pred_test = r_pred_test * 0.8
                test_df_t["propensity_estimate"] = np.clip(p_pred_test, 0.0001, 0.9999)
                test_df_t["relevance_estimate"] = np.clip(r_pred_test, 0.0001, 0.9999)
                test_df_t["treated_estimate"] = t_test_pred
                causal_effect_estimate = \
                    test_df_t["outcome"] * \
                    (test_df_t["treated_estimate"] / test_df_t["propensity_estimate"] - \
                    (1 - test_df_t["treated_estimate"]) / (1 - test_df_t["propensity_estimate"]))
                test_df_t["causal_effect_estimate"] = np.clip(causal_effect_estimate, -1, 1)
                popularity = test_df_t.groupby("idx_item")["idx_user"].nunique().reset_index()
                popularity.columns = ["idx_item", 'popularity']
                test_df_t = test_df_t.merge(popularity, on="idx_item", how='left')
                test_df_t["pred"] = recommender.predict(test_df_t)
                evaluator = Evaluator()
                cp10_tmp_list.append(evaluator.evaluate(test_df_t, 'CPrec', 10))
                cp100_tmp_list.append(evaluator.evaluate(test_df_t, 'CPrec', 100))
                cdcg_tmp_list.append(evaluator.evaluate(test_df_t, 'CDCG', 100000))

                
                ndcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'NDCGR', 10))
                ndcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'NDCGS', 10))
                ndcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'NDCGP', 10))

                recall_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'RecallR', 10))
                recall_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'RecallS', 10))
                recall_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'RecallP', 10))

                precision_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'PrecisionR', 10))
                precision_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'PrecisionR', 10))
                precision_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'PrecisionR', 10))

                kendall_score = evaluator.kendall_tau_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
                spearman_score = evaluator.spearman_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
                pos_diff = evaluator.avg_position_diff(test_df_t, 'idx_user', 'idx_item', 'pred', 'relevance_estimate')

                print(f"Kendall Tau: {kendall_score:.4f}")
                print(f"Spearman Rho: {spearman_score:.4f}")
                print(f"Average Rank Position Difference: {pos_diff:.4f}")

                _ = evaluator.get_sorted(test_df_t)
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
                p_pred_test_t = 0.25 * ((p_pred_test - np.mean(p_pred_test))/ (np.std(p_pred_test)))
                p_pred_test_t = np.clip((p_pred_test_t + 0.5), 0.0, 1.0)

                t_test_pred = np.where(p_pred_test_t >= 0.65, 1.0, 0.0)
                p_pred_test = p_pred_test * 0.2
                r_pred_test = r_pred_test * 0.2
                test_df_t["propensity_estimate"] = np.clip(p_pred_test, 0.0001, 0.9999)
                test_df_t["relevance_estimate"] = np.clip(r_pred_test, 0.0001, 0.9999)
                test_df_t["treated_estimate"] = t_test_pred
                causal_effect_estimate = \
                    test_df_t["outcome"] * \
                    (test_df_t["treated_estimate"] / test_df_t["propensity_estimate"] - \
                    (1 - test_df_t["treated_estimate"]) / (1 - test_df_t["propensity_estimate"]))
                test_df_t["causal_effect_estimate"] = np.clip(causal_effect_estimate, -1, 1)
                popularity = test_df_t.groupby("idx_item")["idx_user"].nunique().reset_index()
                popularity.columns = ["idx_item", 'popularity']
                test_df_t = test_df_t.merge(popularity, on="idx_item", how='left')
                test_df_t["pred"] = recommender.predict(test_df_t)
                evaluator = Evaluator()
                cp10_tmp_list.append(evaluator.evaluate(test_df_t, 'CPrec', 10))
                cp100_tmp_list.append(evaluator.evaluate(test_df_t, 'CPrec', 100))
                cdcg_tmp_list.append(evaluator.evaluate(test_df_t, 'CDCG', 100000))

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

                _ = evaluator.get_sorted(test_df_t)

        cp10 = np.mean(cp10_tmp_list)
        cp100 = np.mean(cp100_tmp_list)
        cdcg = np.mean(cdcg_tmp_list)

        ndcg_rel = np.mean(ndcg_tmp_list_rel)
        ndcg_pred = np.mean(ndcg_tmp_list_pred)
        ndcg_pop = np.mean(ndcg_tmp_list_pop)

        recall_rel = np.mean(recall_tmp_list_rel)
        recall_pred = np.mean(recall_tmp_list_pred)
        recall_pop = np.mean(recall_tmp_list_pop)


        precision_rel = np.mean(precision_tmp_list_rel)
        precision_pred = np.mean(precision_tmp_list_pred)
        precision_pop = np.mean(precision_tmp_list_pop)

        cp10list.append(cp10)
        cp100list.append(cp100)
        cdcglist.append(cdcg)

        ndcglist_rel.append(ndcg_rel)
        ndcglist_pred.append(ndcg_pred)
        ndcglist_pop.append(ndcg_pop)

        recalllist_rel.append(recall_rel)
        recalllist_pred.append(recall_pred)
        recalllist_pop.append(recall_pop)

        precisionlist_rel.append(precision_rel)
        precisionlist_pred.append(precision_pred)
        precisionlist_pop.append(precision_pop)       

    
    with open(plotpath+"/result_" + flag.dataset +".txt", "a+") as f:
        print("CP10:", np.mean(cp10list), np.std(cp10list), file=f)
        print("CP100:", np.mean(cp100list), np.std(cp100list), file=f)
        print("CDCG:", np.mean(cdcglist), np.std(cdcglist), file=f)

        print("NDCG10R:", np.mean(ndcglist_rel), np.std(ndcglist_rel), file=f)
        print("NDCG10S:", np.mean(ndcglist_pred), np.std(ndcglist_pred), file=f)
        print("NDCG10P:", np.mean(ndcglist_pop), np.std(ndcglist_pop), file=f)

        print("Recall10R:", np.mean(recalllist_rel), np.std(recalllist_rel), file=f)
        print("Recall10S:", np.mean(recalllist_pred), np.std(recalllist_pred), file=f)
        print("Recall10P:", np.mean(recalllist_pop), np.std(recalllist_pop), file=f)

        print("Precision10R:", np.mean(precisionlist_rel), np.std(precisionlist_rel), file=f)
        print("Precision10S:", np.mean(precisionlist_pred), np.std(precisionlist_pred), file=f)
        print("Precision10P:", np.mean(precisionlist_pop), np.std(precisionlist_pop), file=f)    

            
if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
    main(flag)
