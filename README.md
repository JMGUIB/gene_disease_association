# PROJECT SUMMARY
With the aim of offering a more personalized medicine, and therefore more effective, to patients, this project will try to generate
a model that makes a prediction of genes-disease association.
## Datasets and alternative data sources
To carry out the following project, the model will be trained using a dataset obtained from the [Therapeutics Data Commons](https://tdcommons.ai/) database, specifically from the following source [Gene-Disease Association](https://tdcommons.ai/multi_pred_tasks/gdi/).
## Data Analysis
In first place we obtain de dataset from the source mentioned before. Then we did a Label Encoder and a get dummies function from Pandas to generate new columns with values of 1 or 0 (True or False) related to the disease and the aminoacid sequence of a gene that is associated or not to the disease. You can check the code on the data_processing&model_definition.ipynb file or in the data_analysis.py file 
## Machine Learning
To be sure of use the best ML model, we revised the bibliography of previous studies, and we realised that the most accurate and the most popular was XGBoost, and for our data set we needed the XGBoostRegressor from sklearn. We trained the model with the parameters: ``` n_estimators = 10, alpha=1, eta=0.2, max_depth=5 ```. You cand check the code on the data_processing&model_definition.ipynb file or in the train.py file 

