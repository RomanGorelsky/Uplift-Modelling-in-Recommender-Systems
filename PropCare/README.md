## How to run:

1. Download from the original dataset from the source and generate the dataset as guided (https://arxiv.org/abs/2008.04563 (DH), https://arxiv.org/abs/2012.09442 (ML), https://github.com/finn-no/recsys_slates_dataset (F)).

2. Modify the path to dataset in `prepare_data` of train.py

3. Execute "python -u main.py --dataset d" for DH_original dataset, "python -u main.py --dataset p" for DH_personalized dataset, "python -u main.py --dataset ml" for ML dataset, "python -u main.py --dataset f" for FINN.no dataset. You can specify whether you want to train the PropCare and DLCE once again by setting "--prop_train true" and "--rec_train true". Also, to select the type of model (original or specified) use "--*_type orig" or "--*_type mod" (where * stands for "prop" or "rec").
