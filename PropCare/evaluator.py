import numpy as np
from scipy.stats import kendalltau
from scipy.stats import spearmanr

class Evaluator():
    def __init__(self,
                 colname_user='idx_user', colname_item='idx_item', colname_time='idx_time',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity',
                 colname_effect='causal_effect', colname_estimate='causal_effect_estimate',
                 colname_relavance = 'relevance_estimate', colname_popularity = 'popularity'):


        self.rank_k = None
        self.colname_user = colname_user
        self.colname_item = colname_item
        self.colname_time = colname_time
        self.colname_outcome = colname_outcome
        self.colname_relavance = colname_relavance
        self.colname_prediction = colname_prediction
        self.colname_treatment = colname_treatment
        self.colname_propensity = colname_propensity
        self.colname_effect = colname_effect
        self.colname_estimate = colname_estimate
        self.colname_popularity = colname_popularity

    def get_ranking(self, df, num_rec=10):
        df = df.sort_values(by=[self.colname_user, self.colname_prediction], ascending=False)
        df_ranking = df.groupby(self.colname_user).head(num_rec)
        if num_rec == 10:
            df_ranking.to_csv('df_ranking_10.csv')
        if num_rec == 100:
            df_ranking.to_csv('df_ranking_100.csv') 
        return df_ranking

    def get_sorted(self, df):
        df = df.sort_values(by=[self.colname_user, self.colname_prediction], ascending=False)
        df.to_csv('df_sorted.csv')
        return df

    def capping(self, df, cap_prop=None):
        if cap_prop is not None and cap_prop > 0:
            bool_cap = np.logical_and(df.loc[:, self.colname_propensity] < cap_prop,
                                      df.loc[:, self.colname_treatment] == 1)
            if np.sum(bool_cap) > 0:
                df.loc[bool_cap, self.colname_propensity] = cap_prop

            bool_cap = np.logical_and(df.loc[:, self.colname_propensity] > 1 - cap_prop,
                                      df.loc[:, self.colname_treatment] == 0)
            if np.sum(bool_cap) > 0:
                df.loc[bool_cap, self.colname_propensity] = 1 - cap_prop

        return df

    def clip(self, df, cap_prop=None):
        if cap_prop is not None and cap_prop > 0:
            pvalue = df[self.colname_propensity].values
            pvalue = np.clip(pvalue, cap_prop, 1-cap_prop)
            df[self.colname_propensity] = pvalue
        return df
    
    def kendall_tau_per_user(self, df, user_col, rank_col_1, rank_col_2):
        taus = []
        for _, group in df.groupby(user_col):
            if len(group) > 1:
                order_1 = group[rank_col_1].rank(ascending=False, method='first')
                print(order_1)
                order_2 = group[rank_col_2].rank(ascending=False, method='first')
                print(order_2)
                tau, _ = kendalltau(order_1, order_2)
                taus.append(tau)
        return np.nanmean(taus)
    

    def spearman_per_user(self,df, user_col, rank_col_1, rank_col_2):
        rhos = []
        for _, group in df.groupby(user_col):
            if len(group) > 1:
                order_1 = group[rank_col_1].rank(ascending=False, method='first')
                order_2 = group[rank_col_2].rank(ascending=False, method='first')
                rho, _ = spearmanr(order_1, order_2)
                rhos.append(rho)
        return np.nanmean(rhos)
    
    def avg_position_diff(self,df, user_col, item_col, pred_col, rel_col):
        diffs = []
        for _, group in df.groupby(user_col):
            group = group.copy()
            group['rank_pred'] = group[pred_col].rank(ascending=False, method='first')
            group['rank_rel'] = group[rel_col].rank(ascending=False, method='first')
            diffs.extend(np.abs(group['rank_pred'] - group['rank_rel']))
        return np.nanmean(diffs)


    def evaluate(self, df_origin, measure, num_rec, mode = 'ASIS', cap_prop=0.0):
        df = df_origin.copy(deep=True)
        # print(df.head())
        df = self.capping(df, cap_prop)
        # df = self.clip(df, cap_prop)
        # print(df.head())
        df = self.get_sorted(df)
        # print(df.head())
        self.rank_k = num_rec

        if 'IPS' in measure:
            df.loc[:, self.colname_estimate] = df.loc[:, self.colname_outcome] * \
                                        (df.loc[:, self.colname_treatment] / df.loc[:,self.colname_propensity] - \
                                         (1 - df.loc[:, self.colname_treatment]) / (1 - df.loc[:, self.colname_propensity]))
        if measure == 'precision':
            return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_outcome: self.prec_at_k})))
        elif measure == 'Prec':
            df_ranking = self.get_ranking(df, num_rec=num_rec)
            return np.nanmean(df_ranking.loc[:, self.colname_outcome].values)
        elif measure == 'CPrec':
            df_ranking = self.get_ranking(df, num_rec=num_rec)
            return np.nanmean(df_ranking.loc[:, self.colname_effect].values)
        elif measure == 'CPrecIPS':
            df_ranking = self.get_ranking(df, num_rec=num_rec)
            return np.nanmean(df_ranking.loc[:, self.colname_estimate].values)
        elif measure == 'DCG':
            return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_outcome: self.dcg_at_k})))
        elif measure == 'CDCG':
            return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_effect: self.dcg_at_k})))
        elif measure == 'CDCGIPS':
            return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_estimate: self.dcg_at_k})))
        elif measure == 'AR':
            return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_outcome: self.ave_rank})))
        elif measure == 'CAR':
            return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_effect: self.ave_rank})))
        elif measure == 'CARIPS':
            return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_estimate: self.ave_rank})))
        elif measure == 'CARP':
            return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_effect: self.arp})))
        elif measure == 'CARPIPS':
            return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_estimate: self.arp})))
        elif measure == 'CARN':
            return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_effect: self.arn})))
        elif measure == 'CARNIPS':
            return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_estimate: self.arn})))
        elif measure == 'NDCG':
            return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_outcome: self.ndcg_at_k})))
        elif measure == 'hit':
            return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_outcome: self.hit_at_k})))
        elif measure == 'AUC':
            return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_outcome: self.auc})))
        elif measure == 'CAUC':
            return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_effect: self.gauc})))
        elif measure == 'CAUCP':
            return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_effect: self.gaucp})))
        elif measure == 'CAUCN':
            return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_effect: self.gaucn})))
        elif measure == 'RecallR':
            recall_scores = df.groupby(self.colname_user).apply(
                lambda x: self.recall_at_k(x, sort_by=self.colname_relavance)
            )
            return float(np.nanmean(recall_scores))
        elif measure == 'RecallP':
            recall_scores = df.groupby(self.colname_user).apply(
                lambda x: self.recall_at_k(x, sort_by=self.colname_popularity)
            )
            return float(np.nanmean(recall_scores))
        elif measure == 'PrecisionS':
            precision_scores = df.groupby(self.colname_user).apply(
                lambda x: self.precision_at_k(x, sort_by=self.colname_prediction)
            )
            return float(np.nanmean(precision_scores))
        elif measure == 'PrecisionR':
            precision_scores = df.groupby(self.colname_user).apply(
                lambda x: self.precision_at_k(x, sort_by=self.colname_relavance)
            )
            return float(np.nanmean(precision_scores))
        elif measure == 'PrecisionP':
            precision_scores = df.groupby(self.colname_user).apply(
                lambda x: self.precision_at_k(x, sort_by=self.colname_popularity)
            )
            return float(np.nanmean(precision_scores))
        elif measure == 'NDCGS':
            ndcg_scores = df.groupby(self.colname_user).apply(
                lambda x: self.ndcg_at_k(x, sort_by=self.colname_prediction, label_col=self.colname_outcome)
            )
            return float(np.nanmean(ndcg_scores))
        elif measure == 'NDCGR':
            ndcg_scores = df.groupby(self.colname_user).apply(
                lambda x: self.ndcg_at_k(x, sort_by=self.colname_relavance, label_col=self.colname_outcome)
            )
            return float(np.nanmean(ndcg_scores))
        elif measure == 'NDCGP':
            ndcg_scores = df.groupby(self.colname_user).apply(
                lambda x: self.ndcg_at_k(x, sort_by=self.colname_popularity, label_col=self.colname_outcome)
            )
            return float(np.nanmean(ndcg_scores))
        else:
            print('measure:"' + measure + '" is not supported! ')


    # functions for each metric
    def prec_at_k(self, x):
        k = min(self.rank_k, len(x))  # rank_k is global variable
        return sum(x[:k]) / k

    def dcg_at_k(self, x):
        k = min(self.rank_k, len(x))  # rank_k is global variable
        return np.sum(x[:k] / np.log2(np.arange(k) + 2))

    def ndcg_at_k(self, x):
        k = min(self.rank_k, len(x))  # rank_k is global variable
        max_dcg_at_k = self.dcg_at_k(sorted(x, reverse=True))
        if max_dcg_at_k == 0:
            return np.nan
        else:
            return self.dcg_at_k(x) / max_dcg_at_k

    def hit_at_k(self, x):
        k = min(self.rank_k, len(x))  # rank_k is global variable
        return float(any(x[:k] > 0))

    def auc(self, x): # for binary (1/0)
        len_x = len(x)
        idx_posi = np.where(x > 0)[0]
        len_posi = len(idx_posi)
        len_nega = len_x - len_posi
        if len_posi == 0 or len_nega == 0:
            return np.nan
        cnt_posi_before_posi = (len_posi * (len_posi - 1)) / 2
        cnt_nega_before_posi = np.sum(idx_posi) - cnt_posi_before_posi
        return 1 - cnt_nega_before_posi / (len_posi * len_nega)

    def gauc(self, x): # AUC with ternary (1/0/-1) value
        x_p = x > 0
        x_n = x < 0
        num_p = np.sum(x_p)
        num_n = np.sum(x_n)
        gauc = 0.0
        if num_p > 0:
            gauc += self.auc(x_p) * (num_p/(num_p + num_n))
        if num_n > 0:
            gauc += (1.0 - self.auc(x_n)) * (num_n/(num_p + num_n))
        return gauc

    def gaucp(self, x):
        return self.auc(x > 0)

    def gaucn(self, x):
        return self.auc(x < 0)

    def ave_rank(self, x):
        len_x = len(x)
        rank = np.arange(len_x) + 1
        return np.mean(x * rank)

    def arp(self, x):
        return self.ave_rank(x > 0)
    def arn(self, x):
        return self.ave_rank(x < 0)

    def dcg_at_k(self, rel_values):
        k = min(self.rank_k, len(rel_values))
        return np.sum(rel_values[:k] / np.log2(np.arange(k) + 2))

    def ndcg_at_k(self, df_user, sort_by, label_col='outcome'):
        df_user = df_user.sort_values(by=sort_by, ascending=False)
        rel = df_user[label_col].values

        k = min(self.rank_k, len(rel))
        ideal_rel = np.sort(rel)[::-1]  # для идеального ранжирования

        dcg = self.dcg_at_k(rel)
        idcg = self.dcg_at_k(ideal_rel)

        return dcg / idcg if idcg > 0 else np.nan
    
    def recall_at_k(self, df_user, sort_by):
        df_user = df_user.sort_values(by=sort_by, ascending=False)
        k = min(self.rank_k, len(df_user))
        rel_in_top_k = df_user['outcome'].iloc[:k].sum()
        total_rel = df_user['outcome'].sum()

        return rel_in_top_k / total_rel if total_rel > 0 else np.nan
    
    def precision_at_k(self, df_user, sort_by):
        df_user = df_user.sort_values(by=sort_by, ascending=False)
        k = min(self.rank_k, len(df_user))
        rel_in_top_k = df_user['outcome'].iloc[:k].sum()
        
        return rel_in_top_k / k if k > 0 else np.nan
