
from datetime import datetime
import numpy as np
import pandas as pd

import random
import os
random.seed(10)

from simulator import DataGeneratorML

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

    parser.add_argument('-ora', '--offset_rating', type=float,
                        default=5.0,
                        help='offset of predicted rating when convert it to the outcome probability under treatement')
    parser.add_argument('-scao', '--scaling_outcome', type=float,
                        default=1.0,
                        help='scaling of predicted watch when convert it to the outcome probability under control')
    parser.add_argument('-scap', '--scaling_propensity', type=float,
                        default=1.0,
                        help='scaling of propensity when based on ranking')
    parser.add_argument('-mas', '--mode_assignment', type=str,
                        help='mode of treatment assignment', default='uniform',
                        required=False)
    parser.add_argument('-nr', '--num_rec', type=int,
                        help='expected number of recommendation', default=210,
                        required=False)
    parser.add_argument('-cap', '--capping', type=float,
                        help='capping', default=0.000001,
                        required=False)

    parser.add_argument('-nc', '--num_CPU', type=int, default=1,
                        help='number of CPU used',
                        required=False)
    parser.add_argument('-ssr', '--set_seed_random', type=int, default=1,
                        help='set seed for randomness',
                        required=False)
    parser.add_argument('-trt', '--trim_train_data', action='store_true',
                        help='remove unpurchased and unrecommended for train data',
                        required=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_arg_parser()
    np.random.seed(seed=args.set_seed_random)

    dir_data_prepared = 'data/synthetic/ML_'+ args.version_of_movielens + '_' + args.mode_assignment + str(args.num_rec) \
                        + '_offset' + format(args.offset_rating, '.1f') + '_scaling' + format(args.scaling_propensity, '.1f') + '/'

    print('dir_data_prepared is {}.'.format(dir_data_prepared))

    if not os.path.exists(dir_data_prepared):
        os.mkdir(dir_data_prepared)

    print('mode_assignment is {}.'.format(args.mode_assignment))

    print('Start prepare data.')
    t_init = datetime.now()

    os.environ['OMP_NUM_THREADS'] = str(args.num_CPU)
    data_generator = DataGeneratorML()

    # load movielens dataset (ML100K, ML25M, etc.)
    data_generator.load_data(args.version_of_movielens)

    # predict rating
    cond_params = args.cond_rating_prediction.split('+')
    dict_params = dict()
    for cond_param in cond_params:
        cond = cond_param.split(':')
        dict_params[cond[0]] = cond[1]
    data_generator.predict_rating(learn_rate=float(dict_params['learn_rate']), iter=int(dict_params['iter']),
                                  dim_factor=int(dict_params['dim_factor']), reg_factor=float(dict_params['reg_factor']))

    # predict observation
    cond_params = args.cond_watch_prediction.split('+')
    dict_params = dict()
    for cond_param in cond_params:
        cond = cond_param.split(':')
        dict_params[cond[0]] = cond[1]
    data_generator.predict_watch(learn_rate=float(dict_params['learn_rate']), iter=int(dict_params['iter']),
                                  dim_factor=int(dict_params['dim_factor']), reg_factor=float(dict_params['reg_factor']))

    # set ground truth probability
    data_generator.set_prob_outcome_treated(offset=args.offset_rating)
    data_generator.set_prob_outcome_control(scaling_outcome=args.scaling_outcome)

    # set propensity
    data_generator.assign_propensity(mode=args.mode_assignment, scaling_propensity=args.scaling_propensity,
                                     num_rec=args.num_rec, capping=args.capping)

    # generate recommendation
    data_generator.assign_treatment()
    # generate outcomes (potential and observed)
    data_generator.assign_outcome()
    # save data
    if args.trim_train_data:
        temp_bool = (data_generator.df_data.loc[:, 'treated'] + data_generator.df_data.loc[:, 'outcome']) > 0
        df_data_train = data_generator.df_data.loc[temp_bool,:]
        df_data_train.to_csv(dir_data_prepared + 'data_train.csv', index=False)
    else:
        data_generator.df_data.to_csv(dir_data_prepared + 'data_train.csv', index=False)

    # vali
    # generate recommendation
    data_generator.assign_treatment()
    # generate outcomes (potential and observed)
    data_generator.assign_outcome()
    # save data

    data_generator.df_data.to_csv(dir_data_prepared + 'data_vali.csv', index=False)

    # test
    # generate recommendation
    data_generator.assign_treatment()
    # generate outcomes (potential and observed)
    data_generator.assign_outcome()
    # save data
    data_generator.df_data.to_csv(dir_data_prepared + 'data_test.csv', index=False)

    print('Data prepared.')
    print('num_users: {}'.format(data_generator.num_users))
    print('num_items: {}'.format(data_generator.num_items))
    print('num_data: {}'.format(data_generator.num_data))
    print(data_generator.df_data.info())

    print('Max propensity: {}'.format(np.max(data_generator.df_data.loc[:, 'propensity'])))
    print('Min propensity: {}'.format(np.min(data_generator.df_data.loc[:, 'propensity'])))
    print('Average propensity: {}'.format(np.mean(data_generator.df_data.loc[:, 'propensity'])))
    print('Average number of recommendations: {}'.format(np.mean(data_generator.df_data.loc[:, 'treated'])*data_generator.num_items))
    print('Ratio of positive outcomes: {}'.format(np.mean(data_generator.df_data.loc[:, 'outcome'])))
    print('Ratio of positive treatment effect: {}'.format(np.mean(data_generator.df_data.loc[:, 'causal_effect'] > 0)))
    print('Ratio of negative treatment effect: {}'.format(np.mean(data_generator.df_data.loc[:, 'causal_effect'] < 0)))
    print('Average treatment effect: {}'.format(np.mean(data_generator.df_data.loc[:, 'causal_effect'])))

    t_end = datetime.now()
    t_diff = t_end - t_init

    hours = (t_diff.days * 24) + (t_diff.seconds / 3600)
    print('Completed in {:.2f} hours.'.format(hours))


