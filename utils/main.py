#Necessary libraries 
import os
import pandas as pd
import numpy as np
import requests
import os

'''Necessary library to obtain the dataset from the database {INSERT DB NAME}'''

#!pip install PyTDC

#Seed specification 

seed = 42
np.random.seed(seed)


from eda import data_engineering
from train import model_train, save_model, load_model, scores

data_engineering()
model_train()
save_model()
load_model()
scores()






