import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
import pandas as pd
import pickle
import numpy as np

global seed
gd = pd.read_csv('path/data_processed.csv', sep = ',')


def model_train():

    #Now we define the X and y vars for further use in Train Test split
    X = gd.drop(['Y'], axis = 1)
    y = gd['Y']

    #Train Test Split
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = seed)

    #Defining model parameters
    #Also if you PC is strong enough you should run a Grid search to find the best parameters
    xgb_r = xg.XGBRegressor(objective ='reg:squarederror', n_estimators = 10, seed = seed, alpha=1, eta=0.2, max_depth=5)

    #Model train
    model = xgb_r.fit(train_X, train_y)

    #Prediction:
    pred = xgb_r.predict(test_X)
    pred
    
    #If we want to save the model with pickel
    import pickle
    with open('/Users/Jose/Desktop/GitHub_Projects/gene_disease_association/src/models/model_V0.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    '''
    #To load the model
    with open('model_V0.pkl', 'rb') as f:
    model = pickle.load(f)
    '''


def save_model():
    
    global model

    '''Save the model with pickel'''
    
    with open('/src/model/model_V0.pkl', 'wb') as f:
        pickle.dump(model, f)

def load_model():

    '''To load the model'''
    with open('/src/model/model_V0.pkl', 'rb') as f:
        model = pickle.load(f)

def scores():
    global test_y, pred, X, y
    
    # Errors
    rmse = np.sqrt(MSE(test_y, pred))
    print(f'MSE: {MSE(test_y, pred)}')
    print(f'MAE: {MAE(test_y, pred)}')
    print("RMSE : % f" %(rmse))