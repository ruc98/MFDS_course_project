
# coding: utf-8

# In[118]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.metrics import cohen_kappa_score
import math


# In[119]:


df = pd.read_csv("IPL_Twitter_MissingData.csv")
df_final= df


# In[120]:


#Dividing the incomplete data into cases depending on the values of Q1, Q2

df_NaN= df.loc[(df['Q1'].isnull()) | (df['Q2'].isnull()) | (df['X1'].isnull()) | (df['X2'].isnull()) | (df['X3'].isnull()) | (df['X4'].isnull())]


df_NaN_NonBinary= df_NaN[['X1','X2','X3','X4']]
A = df_NaN_NonBinary.isnull().sum(axis=1)
A=A.values
df_X1= df_NaN.loc[(df_NaN['X1'].isnull()) & (df_NaN['X2'].isnull()== False)& (df_NaN['X3'].isnull()== False)& (df_NaN['X4'].isnull()== False)]
df_X2= df_NaN.loc[(df_NaN['X2'].isnull()) & (df_NaN['X1'].isnull()== False)& (df_NaN['X3'].isnull()== False)& (df_NaN['X4'].isnull()== False)]
df_X3= df_NaN.loc[(df_NaN['X3'].isnull()) & (df_NaN['X2'].isnull()== False)& (df_NaN['X1'].isnull()== False)& (df_NaN['X4'].isnull()== False)]
df_X4= df_NaN.loc[(df_NaN['X4'].isnull()) & (df_NaN['X2'].isnull()== False)& (df_NaN['X3'].isnull()== False)& (df_NaN['X1'].isnull()== False)]
df_Q1= df_NaN.loc[(df_NaN['Q1'].isnull()) & (df_NaN['Q2'].isnull()== False)& (df_NaN['X1'].isnull()== False)& (df_NaN['X2'].isnull()== False)& (df_NaN['X3'].isnull()== False)& (df_NaN['X4'].isnull()== False)]
df_Q2= df_NaN.loc[(df_NaN['Q2'].isnull()) & (df_NaN['Q1'].isnull()== False)& (df_NaN['X1'].isnull()== False)& (df_NaN['X2'].isnull()== False)& (df_NaN['X3'].isnull()== False)& (df_NaN['X4'].isnull()== False)]


df_X1_b= df_X1.loc[(df_X1['Q1']==0) & (df_X1['Q2']==0)]
df_X1_c= df_X1.loc[(df_X1['Q1']==0) & (df_X1['Q2']==1)]
df_X1_d= df_X1.loc[(df_X1['Q1']==1) & (df_X1['Q2']==0)]
df_X1_e= df_X1.loc[(df_X1['Q1']==1) & (df_X1['Q2']==1)]

df_X2_b= df_X2.loc[(df_X2['Q1']==0) & (df_X2['Q2']==0)]
df_X2_c= df_X2.loc[(df_X2['Q1']==0) & (df_X2['Q2']==1)]
df_X2_d= df_X2.loc[(df_X2['Q1']==1) & (df_X2['Q2']==0)]
df_X2_e= df_X2.loc[(df_X2['Q1']==1) & (df_X2['Q2']==1)]

df_X3_b= df_X3.loc[(df_X3['Q1']==0) & (df_X3['Q2']==0)]
df_X3_c= df_X3.loc[(df_X3['Q1']==0) & (df_X3['Q2']==1)]
df_X3_d= df_X3.loc[(df_X3['Q1']==1) & (df_X3['Q2']==0)]
df_X3_e= df_X3.loc[(df_X3['Q1']==1) & (df_X3['Q2']==1)]

df_X4_b= df_X4.loc[(df_X4['Q1']==0) & (df_X4['Q2']==0)]
df_X4_c= df_X4.loc[(df_X4['Q1']==0) & (df_X4['Q2']==1)]
df_X4_d= df_X4.loc[(df_X4['Q1']==1) & (df_X4['Q2']==0)]
df_X4_e= df_X4.loc[(df_X4['Q1']==1) & (df_X4['Q2']==1)]


df3=df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False) #df3 is same as df2, why to use diff. variables?
df_b= df3.loc[(df3['Q1'] == 0) & (df3['Q2'] == 0) ]
df_b= df_b[['X1','X2','X3','X4']]
df_c= df3.loc[(df3['Q1'] == 0) & (df3['Q2'] == 1) ]
df_c= df_c[['X1','X2','X3','X4']]
df_d= df3.loc[(df3['Q1'] == 1) & (df3['Q2'] == 0) ]
df_d= df_d[['X1','X2','X3','X4']]
df_e= df3.loc[(df3['Q1'] == 1) & (df3['Q2'] == 1) ]
df_e= df_e[['X1','X2','X3','X4']]


# In[121]:



for i in df_X1_b.index:
    df_X1_b.X1.loc[i] = int(round(df_b.X1.mean()))
    
    
for i in df_X1_c.index:
    df_X1_c.X1.loc[i] = int(round(df_c.X1.mean()))
    

for i in df_X1_d.index:
    df_X1_d.X1.loc[i] = int(round(df_d.X1.mean()))
    

for i in df_X1_e.index:
    df_X1_e.X1.loc[i] = int(round(df_e.X1.mean()))
    
    

for i in df_X2_b.index:
    df_X2_b.X2.loc[i] =  int(round(df_b.X2.mean()))
    
    
for i in df_X2_c.index:
    df_X2_c.X2.loc[i] = int(round(df_c.X2.mean()))
    

for i in df_X2_d.index:
    df_X2_d.X2.loc[i] = int(round(df_d.X2.mean()))
    

for i in df_X2_e.index:
    df_X2_e.X2.loc[i] = int(round(df_e.X2.mean()))
    
    
for i in df_X3_b.index:
    df_X3_b.X3.loc[i] =  int(round(df_b.X3.mean()))
    
    
for i in df_X3_c.index:
    df_X3_c.X3.loc[i] = int(round(df_c.X3.mean()))
    

for i in df_X3_d.index:
    df_X3_d.X3.loc[i] = int(round(df_d.X3.mean()))
    

for i in df_X3_e.index:
    df_X3_e.X3.loc[i] = int(round(df_e.X3.mean()))
    

for i in df_X4_b.index:
    df_X4_b.X4.loc[i] =  int(round(df_b.X4.mean()))
    
    
for i in df_X4_c.index:
    df_X4_c.X4.loc[i] = int(round(df_c.X4.mean()))
    

for i in df_X4_d.index:
    df_X4_d.X4.loc[i] = int(round(df_d.X4.mean()))
    

for i in df_X4_e.index:
    df_X4_e.X4.loc[i] = int(round(df_e.X4.mean()))
    
    
#When Both X1 and X2 are unknown.
df_X1_X2 = df_NaN.loc[(df_NaN['X1'].isnull()) & (df_NaN['X2'].isnull())]
df_X1_X2_b= df_X1_X2.loc[(df_X1_X2['Q1']==0) & (df_X1_X2['Q2']==0)]
df_X1_X2_c= df_X1_X2.loc[(df_X1_X2['Q1']==0) & (df_X1_X2['Q2']==1)]
df_X1_X2_d= df_X1_X2.loc[(df_X1_X2['Q1']==1) & (df_X1_X2['Q2']==0)]
df_X1_X2_e= df_X1_X2.loc[(df_X1_X2['Q1']==1) & (df_X1_X2['Q2']==1)]

for i in df_X1_X2_b.index:
    df_X1_X2_b.X1.loc[i] = int(round(df_b.X1.mean()))
    df_X1_X2_b.X2.loc[i] = int(round(df_b.X2.mean()))
    

for i in df_X1_X2_c.index:
    df_X1_X2_c.X1.loc[i] = int(round(df_c.X1.mean()))
    df_X1_X2_c.X2.loc[i] = int(round(df_c.X2.mean()))
    
    
for i in df_X1_X2_d.index:
    df_X1_X2_d.X1.loc[i] = int(round(df_d.X1.mean()))
    df_X1_X2_d.X2.loc[i] = int(round(df_d.X2.mean()))
    
    
for i in df_X1_X2_e.index:
    df_X1_X2_e.X1.loc[i] = int(round(df_e.X1.mean()))
    df_X1_X2_e.X2.loc[i] = int(round(df_e.X2.mean()))
    

    
#When Both X2 and X3 are unknown.
df_X2_X3 = df_NaN.loc[(df_NaN['X2'].isnull()) & (df_NaN['X3'].isnull())]
df_X2_X3_b= df_X2_X3.loc[(df_X2_X3['Q1']==0) & (df_X2_X3['Q2']==0)]
df_X2_X3_c= df_X2_X3.loc[(df_X2_X3['Q1']==0) & (df_X2_X3['Q2']==1)]
df_X2_X3_d= df_X2_X3.loc[(df_X2_X3['Q1']==1) & (df_X2_X3['Q2']==0)]
df_X2_X3_e= df_X2_X3.loc[(df_X2_X3['Q1']==1) & (df_X2_X3['Q2']==1)]

for i in df_X2_X3_b.index:
    df_X2_X3_b.X2.loc[i] = int(round(df_b.X2.mean()))
    df_X2_X3_b.X3.loc[i] = int(round(df_b.X3.mean()))
    

for i in df_X2_X3_c.index:
    df_X2_X3_c.X2.loc[i] = int(round(df_c.X2.mean()))
    df_X2_X3_c.X3.loc[i] = int(round(df_c.X3.mean()))
    
for i in df_X2_X3_d.index:
    df_X2_X3_d.X2.loc[i] = int(round(df_d.X2.mean()))
    df_X2_X3_d.X3.loc[i] = int(round(df_d.X3.mean()))
    
    
for i in df_X2_X3_e.index:
    df_X2_X3_e.X2.loc[i] = int(round(df_e.X2.mean()))
    df_X2_X3_e.X3.loc[i] = int(round(df_e.X3.mean()))
    

    
#When Both X3 and X4 are unknown.
df_X3_X4 = df_NaN.loc[(df_NaN['X3'].isnull()) & (df_NaN['X4'].isnull())]
df_X3_X4_b= df_X3_X4.loc[(df_X3_X4['Q1']==0) & (df_X3_X4['Q2']==0)]
df_X3_X4_c= df_X3_X4.loc[(df_X3_X4['Q1']==0) & (df_X3_X4['Q2']==1)]
df_X3_X4_d= df_X3_X4.loc[(df_X3_X4['Q1']==1) & (df_X3_X4['Q2']==0)]
df_X3_X4_e= df_X3_X4.loc[(df_X3_X4['Q1']==1) & (df_X3_X4['Q2']==1)]

for i in df_X3_X4_b.index:
    df_X3_X4_b.X3.loc[i] = int(round(df_b.X3.mean()))
    df_X3_X4_b.X4.loc[i] = int(round(df_b.X4.mean()))
    

for i in df_X3_X4_c.index:
    df_X3_X4_c.X3.loc[i] = int(round(df_c.X3.mean()))
    df_X3_X4_c.X4.loc[i] = int(round(df_c.X4.mean()))
    
    
for i in df_X3_X4_d.index:
    df_X3_X4_d.X3.loc[i] = int(round(df_d.X3.mean()))
    df_X3_X4_d.X4.loc[i] = int(round(df_d.X4.mean()))
    
    
for i in df_X3_X4_e.index:
    df_X3_X4_e.X3.loc[i] = int(round(df_e.X3.mean()))
    df_X3_X4_e.X4.loc[i] = int(round(df_e.X4.mean()))
    
#Filling missing Q1 values using Naive Bayes

gnb = GaussianNB()
X_test= df_Q1.drop('Q1',axis=1)
y_test= df_Q1.Q1

X_train= df3.drop('Q1', axis=1)
y_train= df3.Q1
gnb.fit(X_train,y_train)

gnb.score(X_train, y_train)

for i in X_test.index:
    df_Q1.Q1.loc[i]= int(round(gnb.predict([X_test.loc[i]])[0])) 
    
    
#Filling missing Q2 values using Naive Bayes

gnb1 = GaussianNB()
X_test= df_Q2.drop('Q2',axis=1)

X_train= df3.drop('Q2', axis=1)
y_train= df3.Q2
gnb1.fit(X_train,y_train)

gnb1.score(X_train, y_train)

for i in X_test.index:
    df_Q2.Q2.loc[i]= int(round(gnb1.predict([X_test.loc[i]])[0]))  



    


# In[122]:


df_X1_concat = pd.concat([df_X1_b, df_X1_c, df_X1_d, df_X1_e])
df_X2_concat = pd.concat([df_X2_b, df_X2_c, df_X2_d, df_X2_e])
df_X3_concat = pd.concat([df_X3_b, df_X3_c, df_X3_d, df_X3_e])
df_X4_concat = pd.concat([df_X4_b, df_X4_c, df_X4_d, df_X4_e])
df_X1_X2_concat = pd.concat([df_X1_X2_b, df_X1_X2_c, df_X1_X2_d, df_X1_X2_e])
df_X2_X3_concat = pd.concat([df_X2_X3_b, df_X2_X3_c, df_X2_X3_d, df_X2_X3_e])
df_X3_X4_concat = pd.concat([df_X3_X4_b, df_X3_X4_c, df_X3_X4_d, df_X3_X4_e])
df_concat= pd.concat([df_X1_concat,df_X2_concat,df_X3_concat,df_X4_concat,df_X1_X2_concat,df_X2_X3_concat,df_X3_X4_concat, df_Q1,df_Q2])
df_imputed= pd.concat([df_concat,df3])


# In[123]:


for i in df_imputed.index:
    df_final.loc[i] = df_imputed.loc[i]
    
print(df_final)

