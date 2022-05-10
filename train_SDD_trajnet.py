#!/usr/bin/env python
# coding: utf-8
import yaml
import pandas as pd
from model import YNet
import datetime

try:
    import pickle5 as pickle
except:
    pass

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

prt_start_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print(prt_start_time)

# #### Some hyperparameters and settings
CONFIG_FILE_PATH = 'config/sdd_trajnet.yaml'  # yaml config file containing all the hyperparameters
DATASET_NAME = 'sdd'

TRAIN_DATA_PATH = 'data/SDD/train_trajnet.pkl'
TRAIN_IMAGE_PATH = 'data/SDD/train'
VAL_DATA_PATH = 'data/SDD/test_trajnet.pkl'
VAL_IMAGE_PATH = 'data/SDD/test'
OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

BATCH_SIZE = 2

# #### Load config file and print hyperparameters
with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]

# #### Load preprocessed Data
try:
    with open(TRAIN_DATA_PATH, "rb") as fh:
        df_train = pickle.load(fh)
    with open(VAL_DATA_PATH, "rb") as fh:
        df_val = pickle.load(fh)
except:
    df_train = pd.read_pickle(TRAIN_DATA_PATH) # in the form of dataframe
    df_val = pd.read_pickle(VAL_DATA_PATH)     # in the form of dataframe
    df_train.head() # return the first n (default n=5) rows

# #### Initiate model and load pretrained weights
model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)

# #### Start training
# Note, the Val ADE and FDE are without TTST and CWS to save time. Therefore, the numbers will be worse than the final values.
model.train(df_train, df_val, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path=VAL_IMAGE_PATH,
            experiment_name=experiment_name, batch_size=BATCH_SIZE, num_goals=NUM_GOALS, num_traj=NUM_TRAJ, 
            device=None, dataset_name=DATASET_NAME)

