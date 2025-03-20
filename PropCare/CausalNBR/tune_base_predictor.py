
from datetime import datetime
import numpy as np
import pandas as pd

import random
import os
random.seed(10)

from simulator import DataGeneratorML
from experimenter import Experimenter

import argparse

def setup_arg_parser():
    parser = argparse.ArgumentParser(
        prog='prepare_data_ml.py',
        usage='prepare semi-synthetic data from movielens',
        description='',
        add_help=True)

    parser.add_argument('-vml', '--version_of_movielens', type=str, default='100k',
                        help='choose a version of movielens', required=False)
    parser.add_argument('-crp', '--cond_rating_prediction', type=str,
                        default='dim_factor:100+reg_factor:0.01+learn_rate:0.001+iter:100000000',
                        help='condition of rating prediction')
    parser.add_argument('-cwp', '--cond_watch_prediction', type=str,
                        default='dim_factor:100+reg_factor:0.01+learn_rate:0.001+iter:100000000',
                        help='condition of watch prediction')
    parser.add_argument('-ts', '--type_search', type=str, default='grid',
                        help='type of search',
                        required=False)
    parser.add_argument('-tot', '--target_of_tuning', type=str,
                        help='rating or watching', default='rating',
                        required=False)
    parser.add_argument('-rv', '--ratio_validation', type=float,
                        help='ratio of validation', default=0.1,
                        required=False)
    parser.add_argument('-ne', '--name_experiment', type=str, default='exp',
                        help='abbreviated name to express the experiment',
                        required=False)

    parser.add_argument('-nc', '--num_CPU', type=int, default=1,
                        help='number of CPU used',
                        required=False)
    parser.add_argument('-ssr', '--set_seed_random', type=int, default=1,
                        help='set seed for randomness',
                        required=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_arg_parser()
    np.random.seed(seed=args.set_seed_random)

    os.environ['OMP_NUM_THREADS'] = str(args.num_CPU)
    data_generator = DataGeneratorML()

    # load movielens dataset (ML100K, ML25M, etc.)
    dir_load = data_generator.load_data(args.version_of_movielens)

    save_result_dir = dir_load + '/result/'
    if not os.path.exists(save_result_dir):
        os.mkdir(save_result_dir)
    save_result_file = save_result_dir + datetime.now().strftime(
        '%Y%m%d_%H%M%S') + "_" + args.target_of_tuning + "_" + args.name_experiment + ".csv"
    print('save_result_file is {}'.format(save_result_file))
    save_result_dir = os.path.dirname(save_result_file)
    if not os.path.exists(save_result_dir):
        os.mkdir(save_result_dir)
    # set common params
    common_params = dict()
    common_params['num_CPU'] = args.num_CPU
    common_params['recommender'] = 'MF'

    if args.target_of_tuning == 'rating':
        experimenter = Experimenter(colname_outcome = 'rating', monitor = ['RMSE'])
        list_params = experimenter.set_search_params(args.cond_rating_prediction, args.type_search)
        common_params['eval_metrics'] = 'RMSE'
        list_params = experimenter.set_common_params(list_params, common_params)
        ind_vali = random.sample(range(data_generator.num_data_raw),
                                 k = int(data_generator.num_data_raw * args.ratio_validation))
        df_vali = data_generator.df_raw.loc[ind_vali, :]
        df_train = data_generator.df_raw.loc[~np.isin(np.arange(data_generator.num_data_raw), ind_vali), :]
        print(df_vali.head(5))
        print(df_train.head(5))
    else:
        experimenter = Experimenter(colname_outcome='watch', monitor=['logloss'])
        list_params = experimenter.set_search_params(args.cond_watch_prediction, args.type_search)
        common_params['eval_metrics'] = 'logloss'
        list_params = experimenter.set_common_params(list_params, common_params)
        experimenter.monitor = ['logloss']
        experimenter.colname_outcome = 'watch'
        ind_vali = random.sample(range(data_generator.num_data),
                                 k=int(data_generator.num_data * args.ratio_validation))
        df_vali = data_generator.df_data.loc[ind_vali, :]
        df_train = data_generator.df_data.loc[~np.isin(np.arange(data_generator.num_data), ind_vali), :]

    print('Start experiment.')
    t_init = datetime.now()
    df_result = experimenter.try_params(list_params, df_train, df_vali, data_generator.num_users, data_generator.num_items,
                                        save_result_file)
    t_end = datetime.now()
    t_diff = t_end - t_init

    hours = t_diff.days * 24 + t_diff.seconds / (60 * 60)

    df_result.to_csv(save_result_file)
    print('Completed in {:.2f} hours.'.format(hours))



