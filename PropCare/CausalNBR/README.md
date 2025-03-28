# Causality-Aware Neighborhood Methods for Recommender Systems

This is our implementation for the paper ["Causality-Aware Neighborhood Methods for Recommender Systems"]

It includes our proposed method *CUBNs/CIBNs* and several baselines (*UBN/IBN*, *ULBPR*, *ULRMF*, *DLCE*, etc.)
It also covers the codes for generating semi-synthetic datasets for the experiments.

## Requirement
The source codes are mostly written in Python.
Only preprocessing the Dunnhumby data requires R.
(Only Python is requiered for MovieLens.)
The libraries requred are listed below.
The versions in parenthesis are our environment for the experiment.

* Python (3.7.0)
* numpy (1.15.1)
* pandas (0.23.4)
* R (3.4.4)
* data.table (1.12.0)

## Usage
The basic usages of our codes with Dunnhumby and MovieLens datasets are as follows. 

### For MovieLens dataset
1. Download the dataset from the MovieLens dataset site.
1. Tune parameters for rating prediction and observation prediction with *tune_base_predictor.py*.
1. Generate a semi-synthetic dataset with *prepare_data_ml.py*.
1. Run the experiment with *param_search_ml.py*.


### For Dunnhumby dataset
1. Download the dataset from the Dunnhumby dataset site.
1. Preprocess the dataset with *preprocess_dunnhumby.R*.
1. Generate a semi-synthetic dataset with *prepare_data.py*.
1. Run the experiment with *param_search.py*.

## For MovieLens dataset
### 1. Download the base dataset
We use ***MovieLens-100K*** and ***MovieLens-1M*** datasets provided by [GroupLens](https://grouplens.org/datasets/movielens).
We really thank GroupLens for providing the various versions of MovieLens datasets that have been standard datasets in recommender system community.
Download the data and locate them under the directory ***./CausalNBR/data/movielens***.



### 2. Preprocess the base dataset 

```bash:sample
cd ./CausalNBR

python tune_base_predictor.py -vml 100k -tot rating -crp num_loop:50+interval_eval:1000000+train_metric:RMSE+dim_factor:100+learn_rate:0.01+reg_factor:0.3:0.1:0.03:0.01+with_bias:False

python tune_base_predictor.py -vml 100k -tot watch -cwp num_loop:100+interval_eval:1000000+train_metric:logloss+dim_factor:100+learn_rate:0.01+reg_factor:0.01:0.001:0.003+with_bias:False
```
The former tunes ratng prediction model that is used in Step 1.
The latter tunes observation (watch movie) prediction model that is used in Step 2.

Regarding some arguments,
- **-vml (version_of_movielens)** specifies the version of MovieLens, e.g., 100k, 1m.
- **-tot (target_of_tuning)** specifies whether we tune rating prediction or watch predictin.
- **-crp (cond_rating_prediction)** specifies exploring conditions for rating prediciton.
- **-cwp (cond_watch_prediction)** specifies exploring conditions for watch prediciton.

After tuning, we obtained best hyper parameter sets for ML-100k as,
```bash:sample
-crp iter:2000000+dim_factor:100+learn_rate:0.01+reg_factor:0.03 -cwp iter:43000000+dim_factor:100+learn_rate:0.01+reg_factor:0.001
```
and for ML-1M as,
```bash:sample
-crp iter:30000000+dim_factor:100+learn_rate:0.01+reg_factor:0.05 -cwp iter:400000000+dim_factor:100+learn_rate:0.01+reg_factor:0.001
```

### 3. Generate a semi-synthetic dataset 
To generate semi-synthetic dataset from ***ML-100k*** dataset by default setting,
```bash:sample
python prepare_data_ml.py -vml 100k -nr 100 -mas logrank -ora 5.0 -scao 1.0 -scap 1.0 -crp iter:2000000+dim_factor:100+learn_rate:0.01+reg_factor:0.03 -cwp iter:43000000+dim_factor:100+learn_rate:0.01+reg_factor:0.001
```
The output folder is *data/preprocessed/data/synthetic/ML_100k_rank100_offset5.0_scaling1.0*.

To generate semi-synthetic dataset from ***ML-1M*** dataset by default setting,
```bash:sample
python prepare_data_ml.py -vml 1m -nr 100 -mas rank -ora 5.0 -scao 1.0 -scap 1.0 -crp iter:30000000+dim_factor:100+learn_rate:0.01+reg_factor:0.05 -cwp iter:400000000+dim_factor:100+learn_rate:0.01+reg_factor:0.001
```
The output folder is *data/preprocessed/data/synthetic/ML_1m_rank100_offset5.0_scaling1.0*.


Regarding some arguments,

- **-ora (offset_rating)** is ***&epsilon;*** in Eq. (11) of Step3. This is set to 5.0.
- **-scap (scaling_propensity)** is ***b*** in Eq. (12) of Step 4. Default is 1.0. We changed this value in the experiments of Fig. 4.
- **-nr (num_rec)** specifies the average number of recommendations for users and it determins ***a*** in Eq. (12) of Step 4. Default is 100. We changed this value in the experiments of Fig. 5.


After the excecution, we obtain three files: *data_train.csv, data_vali_csv, data_test.csv*.
These are data for training, validation, and test as the names suggest.

The *data_train.csv* and *data_test.csv* look like below.
|idx_user|idx_item|treated|outcome|propensity|causal_effect|idx_time|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|0|0|0|0|0.1543|0|0|
|0|1|0|0|0.0154|0|0|
|0|2|0|0|0.1553|0|0|
|0|3|0|0|0.1565|0|0|
|0|4|0|0|0.0937|0|0|
|0|5|0|0|0.0774|0|0|
Note that *data_vali.csv* includes some additional columns for debugging.


### 4. Run the experiment
*param_search_ml.py* conducts experiments with several conditions of hyper-parameters.
The results are saved in **<dataset_folder>/result/<YYYYMMDD_hhmmss_<model_name>_<experiment_name>_tlt1.csv**.
After the parameter tuning, we evaluate on test data by adding argument **-p test**.

See an example below.
```bash:sample
python param_search_ml.py -vml 100k -nr 100 -mas rank -ora 5.0 -scao 1.0 -scap 1.0 -tm CausalNeighborBase -cs num_loop:1+interval_eval:1+way_simil:outcome+measure_simil:cosine+way_neighbor:user+scale_similarity:0.33:0.5:1.0:2.0:3.0+shrinkage:0.0:0.3:1.0:3.0:10.0:30.0:100.0+num_neighbor:1000 -ne CN_UB_nn1000
```

This runs hyper parameter exploration of CUBN-O with combinations of scaling **&alpha; in {0.33, 0.5, 1.0, 2.0, 3.0}** and shrinkage parameter **&beta; in {0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0}** on *ML-100k (b=1.0, number of logged recommendation = 100)*.

Here arguments: **-vml, -nr -mas, -ora, -scao, -scap,**, should be the same with the arguments for corresponding data generation.
Namely, for *ML-1M (b=3.0)* dataset, they become 
```
-vml 1m -nr 100 -mas rank -ora 5.0 -scao 1.0 -scap 3.0
```
**-cs (cond_search)** specifies the hyper-parameter settings.
Parameters are seperated by **+** and explored values are seperated by **:**.

Some notes on parameters,
- **type_model(type_model)** selects a family of methods.
- **way_simil** selects whether similaity is based on outcome (way_simil:outcome) or treatment (way_simil:treatment).
- **measure_simil** selects similarity measure. We used cosine similarity (measure_simil:cosine).
- **way_neighbor** selects whether user-based (way_neighbor:user) or item-based (way_neighbor:item) neighborhood is used. For example, ***way_neighbor:user+way_simil:outcome*** constructs CUBN-O.


The result file includes several metrics and hyperparameters.
**CPrec_10** is **CP@10**, **CPrec_100** is **CP@100**, and **CDCG_100000** is **CDCG**.
Note that the numbers after *_* denote the number of recommendation for evaluation (top-N in the ranking.)
For CDCG, we evaluate the whole ranking, hence the number is intentinally set to be larger than the number of items.

After tuning the parameter, do the test evaluation as following.
```bash:sample
python param_search_ml.py -p test -vml 100k -nr 100 -mas rank -ora 5.0 -scao 1.0 -scap 1.0 -tm CausalNeighborBase -cs num_loop:1+interval_eval:1+way_simil:outcome+measure_simil:cosine+way_neighbor:user+scale_similarity:1.0+shrinkage:1.0+num_neighbor:1000 -ne test_CUBNO
```



## For Dunnhumby dataset
The procedures are mostly same with that described in anc/README.md of ["Unbiased Learning for the Causal Effect of Recommender" (RecSys 2020)](https://arxiv.org/abs/2008.04563)
The difference is that we set ***-tlt 1***, ***-tle 1*** and ***-sf 1.0*** as described in our paper.

### 1. Download the base dataset
We use ***The Complete Journey*** dataset provided by [Dunnhumby](https://www.dunnhumby.com/careers/engineering/sourcefiles).
We really thank Dunnhumby for making this valuable dataset publicly available.
Download the data and locate them under the directory ***./CausalNBR/data/raw***.

### 2. Preprocess the base dataset 

```bash:sample
cd ./CausalNBR
Rscript preprocess_dunnhumby.R
```
It should yield the files named ***cnt_logs.csv*** in both ***.data/preprocessed/dunn_cat_mailer_10_10_1_1/*** and ***.data/preprocessed/dunn_mailer_10_10_1_1/***.
The former is category-level granularity and the latter is product-level granularity.
The obtained files look like below.
|idx_user|idx_item|num_visit|num_treatment|num_outcome|num_treated_outcome|
|:------:|:------:|:------:|:------:|:------:|:------:|
|0|0|66|11|0|0|
|0|1|66|1|0|0|
|0|2|66|11|0|0|
|0|3|66|11|0|0|
|0|4|66|6|1|0|
|0|5|66|5|5|1|
First two columns represent the pair of user-item.
*num_visit* is the number of visit of the user (***&Sigma; V<sub>ut</sub>***).
*num_treatment* is the number of recommendations (***&Sigma; Z<sub>uit</sub>***).
*num_outcome* is the number of purchases (***&Sigma; Y<sub>uit</sub>***).
*num_treated_outcome* is the number of purchases with recommendation (***&Sigma; Z<sub>uit</sub>Y<sub>uit</sub>***).
These are sufficient information for calculating purchase probabilities with and without recommendations for each user-item pair.

### 3. Generate a semi-synthetic dataset 

To generate ***Category-Original*** dataset,
```bash:sample
python prepare_data.py -d data/preprocessed/dunn_cat_mailer_10_10_1_1 -tlt 1 -tlv 1 -tle 1 -rp 0.4 -mas original -cap 0.000001 -trt
```
The output folder is *data/preprocessed/dunn_cat_mailer_10_10_1_1/original_rp0.40*.
Note that we set n_train=1 by *-tlt 1*.

To generate ***Category-Personalized (b=1.0)*** dataset,
```bash:sample
python prepare_data.py -d data/preprocessed/dunn_cat_mailer_10_10_1_1 -tlt 1 -tlv 1 -tle 1 -rp 0.4 -mas rank -sf 1.0 -nr 210 -cap 0.000001 -trt
```
The output folder is *data/preprocessed/dunn_cat_mailer_10_10_1_1/rank_rp0.40_sf1.00_nr210*.
Note that we set &beta;=1.0 as default.

Regarding some arguments,
- **-rp (rate_prior)** specifies the weight of prior ***w***.
- **-sf (scale_factor)** specifies ***b***, so you can obtain datasets with varied uneveness of propensities by changing this value.
- **-nr (num_rec)** specifies the average number of recommendations for users and it determins ***&alpha;***.

After the excecution, we obtain three files: *data_train.csv, data_vali_csv, data_test.csv*.
These are data for training, validation, and test as the names suggest.

The *data_train.csv* and *data_test.csv* looks like below.
|idx_user|idx_item|treated|outcome|propensity|causal_effect|idx_time|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|0|0|0|0|0.1543|0|0|
|0|1|0|0|0.0154|0|0|
|0|2|0|0|0.1553|0|0|
|0|3|0|0|0.1565|0|0|
|0|4|0|0|0.0937|0|0|
|0|5|0|0|0.0774|0|0|
Note that *data_vali.csv* includes some additional columns for debugging.


### 4. Run the experiment
*param_search.py* conducts experiments with several conditions of hyper-parameters.
The results are saved in **<dataset_folder>/result/<YYYYMMDD_hhmmss_<model_name>_<experiment_name>_tlt10.csv**.
After the parameter tuning, we evaluate on test data by adding argument **-p test**.

See an example below.
```bash:sample
python param_search.py -d dunn_cat_mailer_10_10_1_1 -tlt 1 -tle 1 -rp 0.4 -mas rank -sf 1.0 -nr 210 -tm DLMF -cs num_loop:30+interval_eval:20000000+train_metric:AR_logi+dim_factor:200+learn_rate:0.003+naive:False+only_treated:False+capping:0.1:0.01:0.001+reg_factor:0.3:0.03:0.003:0.0003 -ne DLCE_tune_cap_and_reg
```
This runs hyper parameter exploration of DLCE with combinations of capping threshold **&chi; in {0.1, 0.01, 0.001}** and regularization coefficient **&gamma; in {0.3, 0.03, 0.003, 0.0003}** on *Category-Personalized (&beta;=1.0)* dataset.
(This exploration range is simplified for explanation purpose and is different from that described in the paper.)

Here arguments: **-d, -rp, -mas, -sf, -nr**, should be the same with the arguments for corresponding data generation.

**-cs (cond_search)** specifies the hyper-parameter settings.
Parameters are seperated by **+** and explored values are seperated by **:**.

Some notes on parameters,
- **interval_eval** is the number of training iterations between each evaluation and **num_loop** is the number of evaluations. So multiplying them yields total number of training iterations.
- **train_metric** is the training objective and it determins the method together with **-tm** argument. For example, **-DLMF** and **train_metric:AR_logi** is *DLCE* with upper bound loss in Eq. (29), **-DLMF** and **train_metric:AR_sig** is *DLCE* with approximation loss in Eq. (28), **-ULMF** and **train_metric:AUC** is *ULBPR*, **-ULMF** and **train_metric:logloss** is ULRMF, **-LMF** and **train_metric:AUC** is *BPR*.
- **naive:True** select *BLCE* that uses naive estimate in Eq. (12). **naive:False** for *DLCE* and *DLTO*.
- **only_treated:True** select *DLTO* that learns for treated outcomes. **only_treated:False** for *DLCE* and *BLCE*.
- **-ne** describes the name of experiment that is included in the result file name. 


After tuning the parameter, do the test evaluation as following.
```bash:sample
python param_search.py -p test -ce False -d dunn_cat_mailer_10_10_1_1 -tlt 1 -tle 1 -rp 0.4 -mas rank -sf 1.0 -nr 210 -tm DLMF -cs num_loop:1+interval_eval:120000000+train_metric:AR_logi+dim_factor:200+learn_rate:0.003+naive:False+only_treated:False+capping:0.3+reg_factor:0.003 -ne DLCE_test
```

Other examples.

Run *ULBPR* on ***Category-Original*** dataset:
```bash:sample
python param_search.py -d dunn_cat_mailer_10_10_1_1 -tlt 1 -tle 1 -rp 0.4 -mas original -tm ULMF -cs num_loop:30+interval_eval:20000000+train_metric:AUC+dim_factor:200+learn_rate:0.01+alpha:0.0:0.2:0.4+reg_factor:0.3:0.03:0.003:0.0003 -ne ULBPR_tune_alpha_and_reg
```
Note that ULBPR has an unique hyper paramter &alpha; (different from &alpha; in our methods), that is the probability of regarding NR-NP (not recommended and not purchased) items as positives. Please refer to ["Uplift-based Evaluation and Optimization of Recommenders" (RecSys 2019)](https://dl.acm.org/doi/10.1145/3298689.3347018) for the detail.

Run *CausE* (CausE-Prod version) on ***Category-Original*** dataset:
```bash:sample
python param_search.py -d dunn_cat_mailer_10_10_1_1 -tlt 1 -tle 1 -rp 0.4 -mas original -tm CausEProd -cs num_loop:30+interval_eval:20000000+train_metric:logloss+dim_factor:200+learn_rate:0.01+reg_causal:0.1:0.01:0.001+reg_factor:0.3:0.03:0.003:0.0003 -ne CausE_tune_reg
```
Note that CausE has an unique hyper paramter, regularization between tasks (*reg_causal* in our code), that penalizes the difference in item latent factors between treatment and control conditions. Please refer to ["Causal Embedding for Recommendation" (RecSys 2018)](https://dl.acm.org/doi/10.1145/3240323.3240360) for the detail.
We reimplemented CausE to rank items by the causal effect since the original CausE aimed for the prediction of treated outcomes.


## Tuned Hyper-parameters to reproduce the paper.
In case the best parameters are different among evaluation metrics, we chose the best one for each metric.
In practice, the most important metric would differ depending on the actual system and the model should be tuned for the most important metric.
For example, if the number of recommendation is fixed to 10, CP@10 would be appropriate; if the users can view recomendated items as much as they want, CDCG would be appropriate.

Below are tuned hypearparameters for ***ML-100K*** dataset.
|Method|Metric|number of neighbors|scaling factor ***&alpha;***|shrinkage parameter ***&beta;***|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|UBN|CP@10|10|3|0.3|
|UBN|CP@100|1000|0.33|0.3|
|UBN|CDCG|10|3|1|
|UBN|CAR|1000|0.33|0|
|IBN|CP@10|3000|0.33|100|
|IBN|CP@100|3000|0.33|100|
|IBN|CDCG|3000|0.33|0|
|IBN|CAR|3000|0.33|0|
|CUBN-0|CP@10|1000|0.33|30|
|CUBN-0|CP@100|1000|1|3|
|CUBN-0|CDCG|1000|1|3|
|CUBN-0|CAR|1000|2|0.3|
|CUBN-T|CP@10|1000|0.5|30|
|CUBN-T|CP@100|1000|1|3|
|CUBN-T|CDCG|1000|1|3|
|CUBN-T|CAR|1000|3|0.3|
|CIBN-0|CP@10|30|3|0|
|CIBN-0|CP@100|100|0.5|3|
|CIBN-0|CDCG|30|0.33|3|
|CIBN-0|CAR|10|0.5|30|
|CIBN-T|CP@10|300|1|3|
|CIBN-T|CP@100|100|2|0.3|
|CIBN-T|CDCG|300|1|3|
|CIBN-T|CAR|300|2|0.3|
|CUBN-0-woM|CP@10|1000|0.5|3|
|CUBN-0-woM|CP@100|1000|0.33|3|
|CUBN-0-woM|CDCG|1000|0.33|3|
|CUBN-0-woM|CAR|1000|0.33|3|
|CUBN-T-woM|CP@10|1000|0.5|3|
|CUBN-T-woM|CP@100|1000|0.33|3|
|CUBN-T-woM|CDCG|1000|0.33|3|
|CUBN-T-woM|CAR|1000|0.33|3|
|CIBN-0-woM|CP@10|30|0.33|0|
|CIBN-0-woM|CP@100|300|0.33|3|
|CIBN-0-woM|CDCG|300|0.5|3|
|CIBN-0-woM|CAR|10|1|30|
|CIBN-T-woM|CP@10|100|0.5|3|
|CIBN-T-woM|CP@100|300|0.33|10|
|CIBN-T-woM|CDCG|300|0.33|10|
|CIBN-T-woM|CAR|1000|1|3|

|Method|Metric|learning rate ***&eta;***|training iteration (x10<sup>6</sup>)|regularization ***&gamma;***|capping ***&chi;***|NR-NP as positive ***&alpha;***|regularization between tasks|MF dimensions|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|BPR|CP@10|0.001|1.0|0.3|-|-|-|100|
|BPR|CP@100|0.001|1.5|0.3|-|-|-|100|
|BPR|CDCG|0.001|1.5|0.3|-|-|-|100|
|BPR|CAR|0.001|0.1|0.3|-|-|-|100|
|CausE|CP@10|0.003|24|0.01|-|-|0.01|100|
|CausE|CP@100|0.003|76|0.1|-|-|0.1|100|
|CausE|CDCG|0.003|38|0.03|-|-|0.1|100|
|CausE|CAR|0.003|34|0.01|-|-|0.01|100|
|ULRMF|CP@10|0.003|40|0.3|-|0.4|-|100|
|ULRMF|CP@100|0.003|40|0.3|-|0.2|-|100|
|ULRMF|CDCG|0.003|40|0.3|-|0.4|-|100|
|ULRMF|CAR|0.01|400|0.0001|-|1.0|-|100|
|ULBPR|CP@10|0.003|30|0.3|-|0.2|-|100|
|ULBPR|CP@100|0.003|30|0.3|-|0.6|-|100|
|ULBPR|CDCG|0.003|100|0.3|-|0.4|-|100|
|ULBPR|CAR|0.01|300|0.0003|-|0.6|-|100|
|DLTO|CP@10|0.001|6.0|0.3|0.3|-|-|100|
|DLTO|CP@100|0.001|2.5|0.3|0.1|-|-|100|
|DLTO|CDCG|0.001|3.0|0.3|0.1|-|-|100|
|DLTO|CAR|0.001|3.0|0.3|0.1|-|-|100|
|DLCE|CP@10|0.001|30|0.1|0.3|-|-|100|
|DLCE|CP@100|0.001|30|0.3|0.1|-|-|100|
|DLCE|CDCG|0.001|40|0.1|0.3|-|-|100|
|DLCE|CAR|0.001|100|0.1|0.3|-|-|100|

Below are tuned hypearparameters for ***ML-1M*** dataset.
|Method|Metric|number of neighbors|scaling factor ***&alpha;***|shrinkage parameter ***&beta;***|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|UBN|CP@10|10000|0.33|0.3|
|UBN|CP@100|10000|0.33|0.3|
|UBN|CDCG|10000|0.33|0.3|
|UBN|CAR|3000|0.33|0|
|IBN|CP@10|10000|1|0|
|IBN|CP@100|10000|0.5|30|
|IBN|CDCG|10000|0.33|0|
|IBN|CAR|10|0.33|0|
|CUBN-0|CP@10|10000|0.5|30|
|CUBN-0|CP@100|10000|1|3|
|CUBN-0|CDCG|10000|1|3|
|CUBN-0|CAR|10000|2|0.3|
|CUBN-T|CP@10|3000|0.5|30|
|CUBN-T|CP@100|3000|0.5|10|
|CUBN-T|CDCG|3000|1|3|
|CUBN-T|CAR|10|0.33|0|
|CIBN-0|CP@10|30|1|0|
|CIBN-0|CP@100|300|0.33|10|
|CIBN-0|CDCG|10|0.33|1|
|CIBN-0|CAR|10|1|10|
|CIBN-T|CP@10|10|0.33|0|
|CIBN-T|CP@100|300|1|1|
|CIBN-T|CDCG|100|2|0.3|
|CIBN-T|CAR|300|2|0.3|
|CUBN-0-woM|CP@10|10000|0.5|3|
|CUBN-0-woM|CP@100|10000|0.5|3|
|CUBN-0-woM|CDCG|10000|0.5|3|
|CUBN-0-woM|CAR|10000|0.5|3|
|CUBN-T-woM|CP@10|10000|0.33|3|
|CUBN-T-woM|CP@100|10000|0.33|3|
|CUBN-T-woM|CDCG|10000|0.33|3|
|CUBN-T-woM|CAR|10000|0.33|3|
|CIBN-0-woM|CP@10|100|0.33|0|
|CIBN-0-woM|CP@100|30|0.3|0|
|CIBN-0-woM|CDCG|300|0.5|3|
|CIBN-0-woM|CAR|10|1|100|
|CIBN-T-woM|CP@10|300|0.33|10|
|CIBN-T-woM|CP@100|300|0.33|10|
|CIBN-T-woM|CDCG|1000|0.5|10|
|CIBN-T-woM|CAR|1000|1|3|

|Method|Metric|learning rate ***&eta;***|training iteration (x10<sup>6</sup>)|regularization ***&gamma;***|capping ***&chi;***|NR-NP as positive ***&alpha;***|regularization between tasks|MF dimensions|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|BPR|CP@10|0.001|3.4|0.1|-|-|-|100|
|BPR|CP@100|0.001|4.2|0.1|-|-|-|100|
|BPR|CDCG|0.001|2.2|0.3|-|-|-|100|
|BPR|CAR|0.001|0.2|0.03|-|-|-|100|
|CausE|CP@10|0.003|300|0.03|-|-|0.001|100|
|CausE|CP@100|0.003|300|0.03|-|-|0.001|100|
|CausE|CDCG|0.003|300|0.03|-|-|0.001|100|
|CausE|CAR|0.003|400|0.03|-|-|0.0001|100|
|ULRMF|CP@10|0.003|75|0.3|-|0.2|-|100|
|ULRMF|CP@100|0.003|100|0.3|-|0.2|-|100|
|ULRMF|CDCG|0.003|105|0.03|-|0.2|-|100|
|ULRMF|CAR|0.003|110|0.3|-|0.0|-|100|
|ULBPR|CP@10|0.003|75|0.1|-|0.8|-|100|
|ULBPR|CP@100|0.003|60|0.1|-|0.8|-|100|
|ULBPR|CDCG|0.003|75|0.1|-|0.6|-|100|
|ULBPR|CAR|0.003|50|0.03|-|0.2|-|100|
|DLTO|CP@10|0.001|25|0.3|0.3|-|-|100|
|DLTO|CP@100|0.001|15|0.3|0.1|-|-|100|
|DLTO|CDCG|0.001|15|0.3|0.1|-|-|100|
|DLTO|CAR|0.001|15|0.3|0.1|-|-|100|
|DLCE|CP@10|0.001|70|0.1|0.1|-|-|100|
|DLCE|CP@100|0.001|75|0.1|0.1|-|-|100|
|DLCE|CDCG|0.001|100|0.1|0.1|-|-|100|
|DLCE|CAR|0.001|150|0.1|0.1|-|-|100|

Below are tuned hypearparameters for ***Category-Original (tlt=1)*** dataset.
|Method|Metric|number of neighbors|scaling factor ***&alpha;***|shrinkage parameter ***&beta;***|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|UBN|CP@10|3000|0.5|0.3|
|UBN|CP@100|3000|2|0.3|
|UBN|CDCG|1000|0.5|10|
|UBN|CAR|3000|1|0.3|
|IBN|CP@10|1000|2|30|
|IBN|CP@100|1000|2|10|
|IBN|CDCG|1000|1|100|
|IBN|CAR|1000|1|100|
|CUBN-0|CP@10|3000|0.33|10|
|CUBN-0|CP@100|3000|0.33|10|
|CUBN-0|CDCG|3000|1|3|
|CUBN-0|CAR|3000|0.33|0|
|CUBN-T|CP@10|3000|3|1|
|CUBN-T|CP@100|3000|3|3|
|CUBN-T|CDCG|3000|3|1|
|CUBN-T|CAR|3000|3|0|
|CIBN-0|CP@10|30|2|10|
|CIBN-0|CP@100|100|2|100|
|CIBN-0|CDCG|300|3|0.3|
|CIBN-0|CAR|1000|1|3|
|CIBN-T|CP@10|30|3|3|
|CIBN-T|CP@100|30|3|30|
|CIBN-T|CDCG|30|3|3|
|CIBN-T|CAR|30|3|100|
|CUBN-0-woM|CP@10|3000|0.5|3|
|CUBN-0-woM|CP@100|3000|1|3|
|CUBN-0-woM|CDCG|3000|0.33|0|
|CUBN-0-woM|CAR|3000|0.33|0|
|CUBN-T-woM|CP@10|3000|2|1|
|CUBN-T-woM|CP@100|3000|0.5|3|
|CUBN-T-woM|CDCG|3000|0.33|0|
|CUBN-T-woM|CAR|3000|2|1|
|CIBN-0-woM|CP@10|1000|2|100|
|CIBN-0-woM|CP@100|1000|2|100|
|CIBN-0-woM|CDCG|1000|2|100|
|CIBN-0-woM|CAR|1000|1|30|
|CIBN-T-woM|CP@10|1000|0.33|0|
|CIBN-T-woM|CP@100|1000|3|3|
|CIBN-T-woM|CDCG|1000|3|3|
|CIBN-T-woM|CAR|1000|3|3|

|Method|Metric|learning rate ***&eta;***|training iteration (x10<sup>6</sup>)|regularization ***&gamma;***|capping ***&chi;***|NR-NP as positive ***&alpha;***|regularization between tasks|MF dimensions|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|BPR|CP@10|0.003|5|0.1|-|-|-|200|
|BPR|CP@100|0.003|43|0.3|-|-|-|200|
|BPR|CDCG|0.003|18|0.0001|-|-|-|200|
|BPR|CAR|0.003|18|0.0001|-|-|-|200|
|CausE|CP@10|0.01|120|0.03|-|-|0.01|200|
|CausE|CP@100|0.01|60|0.01|-|-|0.01|200|
|CausE|CDCG|0.01|60|0.01|-|-|0.01|200|
|CausE|CAR|0.01|230|0.03|-|-|0.001|200|
|ULRMF|CP@10|0.01|95|0.01|-|0.0|-|200|
|ULRMF|CP@100|0.01|10|0.03|-|0.2|-|200|
|ULRMF|CDCG|0.01|50|0.03|-|0.0|-|200|
|ULRMF|CAR|0.01|15|0.3|-|0.0|-|200|
|ULBPR|CP@10|0.01|190|0.1|-|0.6|-|200|
|ULBPR|CP@100|0.01|190|0.3|-|0.2|-|200|
|ULBPR|CDCG|0.01|40|0.1|-|0.0|-|200|
|ULBPR|CAR|0.01|40|0.1|-|0.0|-|200|
|DLTO|CP@10|0.003|40|0.0003|0.003|-|-|200|
|DLTO|CP@100|0.003|40|0.3|0.3|-|-|200|
|DLTO|CDCG|0.003|50|0.1|0.001|-|-|200|
|DLTO|CAR|0.003|170|0.3|0.3|-|-|200|
|DLCE|CP@10|0.003|10|0.0001|0.001|-|-|200|
|DLCE|CP@100|0.003|200|0.3|0.3|-|-|200|
|DLCE|CDCG|0.003|10|0.0003|0.01|-|-|200|
|DLCE|CAR|0.003|200|0.3|0.3|-|-|200|


Below are tuned hyparparameters for ***Category-Personalized (b=1.0, tlt=1)*** dataset.
|Method|Metric|number of neighbors|scaling factor ***&alpha;***|shrinkage parameter ***&beta;***|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|UBN|CP@10|100|1|0.3|
|UBN|CP@100|30|0.33|0|
|UBN|CDCG|10|3|0|
|UBN|CAR|10|3|0|
|IBN|CP@10|300|0.5|100|
|IBN|CP@100|1000|1|30|
|IBN|CDCG|1000|1|30|
|IBN|CAR|10|2|0.3|
|CUBN-0|CP@10|300|3|1|
|CUBN-0|CP@100|1000|1|3|
|CUBN-0|CDCG|3000|3|0.3|
|CUBN-0|CAR|3000|1|0.3|
|CUBN-T|CP@10|3000|5|0|
|CUBN-T|CP@100|3000|3|0.3|
|CUBN-T|CDCG|3000|5|0|
|CUBN-T|CAR|3000|3|0|
|CIBN-0|CP@10|1000|1|3|
|CIBN-0|CP@100|1000|2|100|
|CIBN-0|CDCG|1000|2|3|
|CIBN-0|CAR|10|3|10|
|CIBN-T|CP@10|1000|2|3|
|CIBN-T|CP@100|1000|2|3|
|CIBN-T|CDCG|1000|2|3|
|CIBN-T|CAR|10|3|100|
|CUBN-0-woM|CP@10|1000|2|0.3|
|CUBN-0-woM|CP@100|3000|1|0|
|CUBN-0-woM|CDCG|3000|0.5|0|
|CUBN-0-woM|CAR|10|1|3|
|CUBN-T-woM|CP@10|3000|3|3|
|CUBN-T-woM|CP@100|3000|2|0|
|CUBN-T-woM|CDCG|3000|3|0|
|CUBN-T-woM|CAR|10|0.33|0|
|CIBN-0-woM|CP@10|10|2|0.3|
|CIBN-0-woM|CP@100|1000|2|3|
|CIBN-0-woM|CDCG|10|2|100|
|CIBN-0-woM|CAR|10|3|0.3|
|CIBN-T-woM|CP@10|1000|3|1|
|CIBN-T-woM|CP@100|300|3|1|
|CIBN-T-woM|CDCG|1000|3|3|
|CIBN-T-woM|CAR|10|3|1|

|Method|Metric|learning rate ***&eta;***|training iteration (x10<sup>6</sup>)|regularization ***&gamma;***|capping ***&chi;***|NR-NP as positive ***&alpha;***|regularization between tasks|MF dimensions|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|BPR|CP@10|0.003|150|0.001|-|-|-|400|
|BPR|CP@100|0.003|200|0.03|-|-|-|400|
|BPR|CDCG|0.003|55|0.1|-|-|-|400|
|BPR|CAR|0.003|100|0.03|-|-|-|400|
|CausE|CP@10|0.01|330|0.0003|-|-|0.0001|400|
|CausE|CP@100|0.01|60|0.01|-|-|0.0001|400|
|CausE|CDCG|0.01|400|0.003|-|-|0.001|400|
|CausE|CAR|0.01|60|0.01|-|-|0.0001|400|
|ULRMF|CP@10|0.01|150|0.01|-|0.6|-|400|
|ULRMF|CP@100|0.01|60|0.03|-|0.2|-|400|
|ULRMF|CDCG|0.01|50|0.01|-|0.0|-|400|
|ULRMF|CAR|0.01|100|0.03|-|0.0|-|400|
|ULBPR|CP@10|0.01|200|0.03|-|0.4|-|400|
|ULBPR|CP@100|0.01|30|0.03|-|0.0|-|400|
|ULBPR|CDCG|0.01|100|0.03|-|0.2|-|400|
|ULBPR|CAR|0.01|30|0.1|-|0.0|-|400|
|DLTO|CP@10|0.003|140|0.003|0.01|-|-|400|
|DLTO|CP@100|0.003|50|0.03|0.003|-|-|400|
|DLTO|CDCG|0.003|60|0.003|0.3|-|-|400|
|DLTO|CAR|0.003|80|0.03|0.7|-|-|400|
|DLCE|CP@10|0.003|100|0.001|0.1|-|-|400|
|DLCE|CP@100|0.003|30|0.003|0.7|-|-|400|
|DLCE|CDCG|0.003|260|0.0003|0.7|-|-|400|
|DLCE|CAR|0.003|12|0.1|0.3|-|-|400|
