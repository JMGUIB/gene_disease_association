#Data Process file
import pandas as pd
import os

#Data_collection:
    
#Gene-Disease Association 
from tdc.multi_pred import GDA
gd_data = GDA(name = 'DisGeNET') #This var has all the dataset from de DB and create a directory called data

raw_data = pd.read_csv('/Users/Jose/Desktop/The_Bridge/Core/3-Machine_Learning/ml_pro/src/data/disgenet.csv') #Raw data to csv

    
def data_engineering():

    #Necessary library to obtain the dataset from the database Terapeutic Data Commons
    #We only have to do install it if we don't already have it

    #!pip install PyTDC
    #To download the data into the data directory we have to change to that directory
    #os.chdir('path')
    #TDC provides us the command line to download the data

    '''Data Collection'''
    #Gene-Disease Association 
    from tdc.multi_pred import GDA
    gd_data = GDA(name = 'DisGeNET')

    #We can download the data already splitted into train and test, but in this case, we are going to do it by ourselves
    #However, here is the command line to download the splitted data
    #split = gd_data.get_split()
    '''The TDC DB supplies us the raw data already "processed"'''
    #Data to DataFrame
    df = gd_data.get_data()
    df.head(3)  #As we can see, we have 4 columns Gene_ID, Gene (coded into the one letter or IUPAC code), Disease_ID, 
                #Disease (with the disease's name and a brief description), and Y (values from 0 to 1)
                
    #To not adulterate the original data, we create a copy called df 
    gd = df.copy()
    ### Data processing 
    #We change the Disease column to lowercase:

    gd['Disease'] = gd['Disease'].str.lower()
    gd.head()
    #Now we create a list wit the aim to separate the disease's name from the description and we split it from the ':'

    dis_list = gd['Disease'].str.split(':')
    dis_list.head(3)
    #The next step is to modify the original Disease column which has the name and the description, replacing it with just the disease's name:
    index = 0
    name_list = [] 
    for index,dis in enumerate(dis_list):
        dis = dis_list[index][0]
        name_list.append(dis)
        index += 1

    #Finally we can add the name into the dataframe
    gd['Disease'] = name_list
    gd.head(3)
    #Also we have to apply a Label Encoder from sklearn to transform non-numerical labels to numerical
    from sklearn import preprocessing

    le = preprocessing.LabelEncoder()
    gd['Gene'] = le.fit_transform(gd.Gene.values)
    gd['Disease'] = le.fit_transform(gd.Disease.values)
    #For a proper data analysis we can run a get_dummies to Gene, and Disease columns and join them to the gd dataframe
    #Get Dummies to dataframe
    seq = pd.get_dummies(gd['Gene'])
    dis_name = pd.get_dummies(gd['Disease_ID'])
    gd = gd.join(seq)
    gd = gd.join(dis_name)

    #As we won't need some of original columns in this dataframe, we can drop them out, except the Y column, we keep that one

    gd = gd.drop(['Gene_ID', 'Disease_ID'], axis = 1)
    gd.head(3)
    #We make sure that all of data is numeric
    gd.info()

    #Now we save the processed data
    gd.to_csv('path/processed_data.csv', sep=',', index=False)
