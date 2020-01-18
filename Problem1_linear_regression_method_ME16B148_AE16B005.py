### Problem01, Regression Method ###

import numpy as np
import pandas as pd
import os
import sys

full_data = pd.read_csv('IPL_Twitter_MissingData.csv')
df = full_data.dropna(0,how="any")  # training data is 762 rows which are complete


varlist = ['Q1','Q2','X1','X2','X3','X4']   # defining the list of variables

# this is loop for 1 missing variable, the training data is 5 dimensional data of rest 5 variables
for var in varlist:         # loop for iterating over variables to fill values using regression
    list2 = varlist.copy()
    list2.remove(var)       # list of variables except variable to be predicted
    X = df[list2].values
    Y = df[var].values
    W = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)  # weights for linear regression
    for index,row in full_data.iterrows():                  # iteration over rows
        if (pd.isnull(row[var]) == True) and (pd.isnull(row[list2]).any()!=True):   # if only prediction variable is nan and rest are not nan
            x = np.array(row[list2])
            if var=='Q1' or var=='Q2':              # if 'Q1' or 'Q2', then set values<0.5 as 0 and rest as 1
                if np.dot(W.T,x)<0.5: row[var]=0
                else: row[var]=1
            else:
                row[var] = int(np.dot(W.T,x))


 

# this is loop for 2 missing variables, the training data is rest 4 variables.
for var1 in varlist:
    for var2 in varlist:
        if var1==var2:
            continue
        else:
            list2 = varlist.copy()
            list2.remove(var1)
            list2.remove(var2)
            X  = df[list2].values
            Y1 = df[var1].values
            Y2 = df[var2].values
            W1 = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y1)    # two models, W1 to predict var1 and W2 to predict var2
            W2 = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y2)
            for index,row in full_data.iterrows():
                if (pd.isnull(row[var1]) == True) and (pd.isnull(row[var2]) == True) and (pd.isnull(row[list2]).any()!=True):
                    x = np.array(row[list2])
                    if var1=='Q1' or var1=='Q2':
                        if np.dot(W1.T,x)<0.5: row[var1]=0
                        else: row[var1]=1
                    else:
                        row[var1] = int(np.dot(W1.T,x))
                    if var2=='Q1' or var2=='Q2':
                        if np.dot(W2.T,x)<0.5: row[var2]=0
                        else: row[var2]=1
                    else:
                        row[var2] = int(np.dot(W2.T,x))


 


vect = full_data.isna().sum(1).sum(0)
#print('Total no of nan values is ',vect)
print(full_data)
full_data.to_csv('Imputed Data Approach2.csv')
print('Hence, all the Missing data is filled')


 




