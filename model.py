'''
Working on Gradient Boosting technique to identify feature importances
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
fname = 'data/Train.csv'

def load_data():
    ''' 
    Loading data from csv file and returning a pandas data frame
    '''
    data = pd.read_csv(fname)
    y = data.pop('SalePrice')
    salesid = data.pop('SalesID')
    data['saledate'] = pd.to_datetime(data['saledate'])
    data['age'] = data['saledate'].dt.year - data['YearMade']
    X = data[['datasource', 'auctioneerID', 'MachineHoursCurrentMeter', 'UsageBand', 'fiModelDesc', 'ProductSize', 'ProductGroup', 'Enclosure']]
    return X, y

def dummy_col(data, feature):
    '''
    Modifies the original data frame
    Input - pandas data frame and feature/column you want to dummify
    Output - None
    '''
    data[feature].fillna('missing',inplace= True )
    for elem in data[feature].unique():
        data[str(feature)+str(elem)] = data[feature] == elem

def clean(data):
    '''
    Cleaning a datagrame's non continuous features
    Input - Pandas Data Frame which requires to be cleaned
    Output - None 
    '''
    z = data.columns
    for i in z:
        if data[i].dtype != 'int64':
            dummy_col(data, i)

def clean_data(df):
    '''
    Input - Data Frame to clean  
    Output - 
    '''
    df["saledate"] = pd.to_datetime(df['saledate'])
    df['saleyear'] = df['saledate'].dt.year
    df["Age"] = df['saleyear'] - df["YearMade"]
    categories = df[["Enclosure", "ProductSize"]]
    continuous = df["Age", "MachineHoursCurrentMeter"]
    for k in categories :
        categories[k] = categories[k].apply(checkMissing)
    continuous["MachineHoursCurrentMeter"] = continuous["MachineHoursCurrentMeter"].apply(checkMissing)
    for i in categories.columns:
        x = pd.get_dummies(categories[i], prefix=i)
        continuous = pd.concat([continuous, x],axis=1)
    for i in continuous.columns:
        bins =  np.array([continuous[i].quantile(x) for x in np.arange(0,1,0.1)])
        continuous[i+'binned'] = pd.cut(continuous[i], bins, retbins= True)
        x = pd.get_dummies(continuous[i +'binned'], prefix=i)
        continuous = pd.concat([continuous, x],axis=1)
    continuous["Age_missing"] = continuous["Age"].apply(dummy_age)
    continuous["machinehourscurrentmeter_missing"] = continuous["MachineHoursCurrentMeter"].apply(dummy_age)
    return continuous


if __name__ == '__main__':
    X, y= load_data()
    clean_data(X)
    model = GradientBoostingRegressor()
    lm = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scoring = make_scorer(accuracy_score)
    res = cross_val_score(estimator=model, X=X_train, y = y_train, scoring=scoring,  cv= 5, n_jobs=1)
    res_lr = cross_val_score(estimator=lm, X=X_train, y = y_train, scoring=scoring,  cv= 5, n_jobs=1)
    
