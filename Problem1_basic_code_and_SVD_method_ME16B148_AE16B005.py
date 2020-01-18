
# coding: utf-8

# In[318]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import math


# In[319]:


df = pd.read_csv("IPL_Twitter_MissingData.csv")
df_original= df


# In[320]:


Total_No_of_Nan=df.isnull().sum()
No_of_DataSamples_with_Nan=df.isnull().T.any().T.sum()
print('Number of Data Samples with NaN =', No_of_DataSamples_with_Nan)


# In[321]:


df1=df[['X1','X2','X3','X4']]
df2 = df1.dropna()
#print(df2.isnull().sum())
#print(df2.isnull().T.any().T.sum())


# In[322]:


U1,S1,VT1 = np.linalg.svd(scale(df2))
#print(S1)
#print(VT1)
vector_a = VT1[3,:]



df3=df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False) 
df_b= df3.loc[(df3['Q1'] == 0) & (df3['Q2'] == 0) ]
df_b= df_b[['X1','X2','X3','X4']]
df_c= df3.loc[(df3['Q1'] == 0) & (df3['Q2'] == 1) ]
df_c= df_c[['X1','X2','X3','X4']]
df_d= df3.loc[(df3['Q1'] == 1) & (df3['Q2'] == 0) ]
df_d= df_d[['X1','X2','X3','X4']]
df_e= df3.loc[(df3['Q1'] == 1) & (df3['Q2'] == 1) ]
df_e= df_e[['X1','X2','X3','X4']]
df_bb= (df_b-df_b.mean())
df_cc= (df_c-df_c.mean())
df_dd= (df_d-df_d.mean())
df_ee= (df_e-df_e.mean())
c = np.zeros(4)

U2,S2,VT2 = np.linalg.svd((df_bb))
U3,S3,VT3 = np.linalg.svd((df_cc))
U4,S4,VT4 = np.linalg.svd((df_dd))
U5,S5,VT5 = np.linalg.svd((df_ee))
vector_b = VT2[3,:]
vector_c = VT3[3,:]
vector_d = VT4[3,:]
vector_e = VT5[3,:]
c[0] = vector_b.dot(df_b.mean())
c[1] = vector_c.dot(df_c.mean())
c[2] = vector_d.dot(df_d.mean())
c[3] = vector_e.dot(df_e.mean())

print('Linear Relationship Cofficient between different variables for case A',vector_a)
print('Linear Relationship Cofficient between different variables for case B',vector_b)
print('Linear Relationship Cofficient between different variables for case C',vector_c)
print('Linear Relationship Cofficient between different variables for case D',vector_d)
print('Linear Relationship Cofficient between different variables for case E',vector_e)



# In[323]:


df_NaN= df.loc[(df['Q1'].isnull()) | (df['Q2'].isnull()) | (df['X1'].isnull()) | (df['X2'].isnull()) | (df['X3'].isnull()) | (df['X4'].isnull())]
#print(df_NaN)


# In[324]:


#Dividing the incomplete data into cases depending on the values of Q1, Q2


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




# In[325]:


df_X1_b= df_X1_b[['X1','X2','X3','X4']]
df_X1_b= (df_X1_b - df_b.mean())
for i in df_X1_b.index:
    df_X1_b.X1.loc[i] = ((-vector_b[1:4].dot(df_X1_b.loc[i][1:5]))/vector_b[0])
    df_X1_b.X1.loc[i] = round(df_X1_b.X1.loc[i] + df_b.mean()['X1'])
    df_X1_b.X2.loc[i]= df_X1_b.X2.loc[i] + df_b.mean()['X2']
    df_X1_b.X3.loc[i]= df_X1_b.X3.loc[i] + df_b.mean()['X3']
    df_X1_b.X4.loc[i]= df_X1_b.X4.loc[i] + df_b.mean()['X4']
    #print(df_X1_b.X1[i])
    
df_X1_c= df_X1_c[['X1','X2','X3','X4']]
df_X1_c= (df_X1_c - df_c.mean())    
for i in df_X1_c.index:
    df_X1_c.X1.loc[i] = ((-vector_c[1:4].dot(df_X1_c.loc[i][1:5]))/vector_c[0])
    df_X1_c.X1.loc[i] = round(df_X1_c.X1.loc[i] + df_c.mean()['X1'])
    df_X1_c.X2.loc[i]= df_X1_c.X2.loc[i] + df_c.mean()['X2']
    df_X1_c.X3.loc[i]= df_X1_c.X3.loc[i] + df_c.mean()['X3']
    df_X1_c.X4.loc[i]= df_X1_c.X4.loc[i] + df_c.mean()['X4']
    #print(df_X1_c.X1[i])

df_X1_d= df_X1_d[['X1','X2','X3','X4']]
df_X1_d= (df_X1_d - df_d.mean())
for i in df_X1_d.index:
    df_X1_d.X1.loc[i] = ((-vector_d[1:4].dot(df_X1_d.loc[i][1:5]))/vector_d[0])
    df_X1_d.X1.loc[i] = round(df_X1_d.X1.loc[i] + df_d.mean()['X1'])
    df_X1_d.X2.loc[i]= df_X1_d.X2.loc[i] + df_d.mean()['X2']
    df_X1_d.X3.loc[i]= df_X1_d.X3.loc[i] + df_d.mean()['X3']
    df_X1_d.X4.loc[i]= df_X1_d.X4.loc[i] + df_d.mean()['X4']
    #print(df_X1_d.X1[i])

df_X1_e= df_X1_e[['X1','X2','X3','X4']]  
df_X1_e= (df_X1_e - df_e.mean())    
for i in df_X1_e.index:
    df_X1_e.X1.loc[i] = ((-vector_e[1:4].dot(df_X1_e.loc[i][1:5]))/vector_e[0])
    df_X1_e.X1.loc[i] = round(df_X1_e.X1.loc[i] + df_e.mean()['X1'])
    df_X1_e.X2.loc[i]= df_X1_e.X2.loc[i] + df_e.mean()['X2']
    df_X1_e.X3.loc[i]= df_X1_e.X3.loc[i] + df_e.mean()['X3']
    df_X1_e.X4.loc[i]= df_X1_e.X4.loc[i] + df_e.mean()['X4']
    #print(df_X1_e.X1[i])


# In[326]:


#When only X2 is missing

df_X2_b= df_X2_b[['X1','X2','X3','X4']]
df_X2_b= (df_X2_b - df_b.mean())
for i in df_X2_b.index:
    df_X2_b.X2.loc[i] = ((-np.delete(vector_b,1).dot(df_X2_b.loc[i].drop('X2')[0:4]))/vector_b[2])
    df_X2_b.X2.loc[i] = round(df_X2_b.X2.loc[i] + df_b.mean()['X2'])
    df_X2_b.X1.loc[i]= df_X2_b.X1.loc[i] + df_b.mean()['X1']
    df_X2_b.X3.loc[i]= df_X2_b.X3.loc[i] + df_b.mean()['X3']
    df_X2_b.X4.loc[i]= df_X2_b.X4.loc[i] + df_b.mean()['X4']
    #print(df_X2_b.X2[i])
    
df_X2_c= df_X2_c[['X1','X2','X3','X4']]
df_X2_c= (df_X2_c - df_c.mean())
for i in df_X2_c.index:
    df_X2_c.X2.loc[i] = ((-np.delete(vector_c,1).dot(df_X2_c.loc[i].drop('X2')[0:4]))/vector_c[2])
    df_X2_c.X2.loc[i] = round(df_X2_c.X2.loc[i] + df_c.mean()['X2'])
    df_X2_c.X1.loc[i]= df_X2_c.X1.loc[i] + df_c.mean()['X1']
    df_X2_c.X3.loc[i]= df_X2_c.X3.loc[i] + df_c.mean()['X3']
    df_X2_c.X4.loc[i]= df_X2_c.X4.loc[i] + df_c.mean()['X4']
    #print(df_X2_c.X2[i])

df_X2_d= df_X2_d[['X1','X2','X3','X4']]
df_X2_d= (df_X2_d - df_d.mean())
for i in df_X2_d.index:
    df_X2_d.X2.loc[i] = ((-np.delete(vector_d,1).dot(df_X2_d.loc[i].drop('X2')[0:4]))/vector_d[2])
    df_X2_d.X2.loc[i] = round(df_X2_d.X2.loc[i] + df_d.mean()['X2'])
    df_X2_d.X1.loc[i]= df_X2_d.X1.loc[i] + df_d.mean()['X1']
    df_X2_d.X3.loc[i]= df_X2_d.X3.loc[i] + df_d.mean()['X3']
    df_X2_d.X4.loc[i]= df_X2_d.X4.loc[i] + df_d.mean()['X4']
    #print(df_X2_d.X2[i])
   
df_X2_e= df_X2_e[['X1','X2','X3','X4']]
df_X2_e= (df_X2_e - df_e.mean())
for i in df_X2_e.index:
    df_X2_e.X2.loc[i] = ((-np.delete(vector_e,1).dot(df_X2_e.loc[i].drop('X2')[0:4]))/vector_e[2])
    df_X2_e.X2.loc[i] = round(df_X2_e.X2.loc[i] + df_e.mean()['X2'])
    df_X2_e.X1.loc[i]= df_X2_e.X1.loc[i] + df_e.mean()['X1']
    df_X2_e.X3.loc[i]= df_X2_e.X3.loc[i] + df_e.mean()['X3']
    df_X2_e.X4.loc[i]= df_X2_e.X4.loc[i] + df_e.mean()['X4']
    #print(df_X2_e.X2[i])


# In[327]:


#When only X3 is missing

df_X3_b= df_X3_b[['X1','X2','X3','X4']]
df_X3_b= (df_X3_b - df_b.mean())
for i in df_X3_b.index:
    df_X3_b.X3.loc[i] = ((-np.delete(vector_b,2).dot(df_X3_b.loc[i].drop('X3')))/vector_b[2])
    df_X3_b.X3.loc[i] = round(df_X3_b.X3.loc[i] + df_b.mean()['X3'])
    df_X3_b.X1.loc[i]= df_X3_b.X1.loc[i] + df_b.mean()['X1']
    df_X3_b.X2.loc[i]= df_X3_b.X2.loc[i] + df_b.mean()['X2']
    df_X3_b.X4.loc[i]= df_X3_b.X4.loc[i] + df_b.mean()['X4']
    #print(df_X3_b.X3[i])
    
df_X3_c= df_X3_c[['X1','X2','X3','X4']]
df_X3_c= (df_X3_c - df_c.mean())
for i in df_X3_c.index:
    df_X3_c.X3.loc[i] = ((-np.delete(vector_c,2).dot(df_X3_c.loc[i].drop('X3')))/vector_c[2])
    df_X3_c = round(df_X3_c + df_c.mean())
    #print(df_X3_c.X3[i])

df_X3_d= df_X3_d[['X1','X2','X3','X4']]
df_X3_d= (df_X3_d - df_d.mean())
for i in df_X3_d.index:
    df_X3_d.X3.loc[i] = ((-np.delete(vector_d,2).dot(df_X3_d.loc[i].drop('X3')))/vector_d[2])
    df_X3_d = round(df_X3_d + df_d.mean())
    #print(df_X3_d.X3[i])
   
df_X3_e= df_X3_e[['X1','X2','X3','X4']]
df_X3_e= (df_X3_e - df_e.mean())
for i in df_X3_e.index:
    df_X3_e.X3.loc[i] = ((-np.delete(vector_e,2).dot(df_X3_e.loc[i].drop('X3')))/vector_e[2])
    df_X3_e = round(df_X3_e + df_e.mean())
    #print(df_X3_e.X3[i])


# In[328]:


# When only X4 is missing

df_X4_b= df_X4_b[['X1','X2','X3','X4']]
df_X4_b= (df_X4_b - df_b.mean())
for i in df_X4_b.index:
    df_X4_b.X4.loc[i] = ((-np.delete(vector_b,3).dot(df_X4_b.loc[i].drop('X4')))/vector_b[3])
    df_X4_b.X4.loc[i] = round(df_X4_b.X4.loc[i] + df_b.mean()['X4'])
    df_X4_b.X1.loc[i]= df_X4_b.X1.loc[i] + df_b.mean()['X1']
    df_X4_b.X2.loc[i]= df_X4_b.X2.loc[i] + df_b.mean()['X2']
    df_X4_b.X4.loc[i]= df_X4_b.X3.loc[i] + df_b.mean()['X3']
    #print(df_X4_b.X4[i])
    
df_X4_c= df_X4_c[['X1','X2','X3','X4']]
df_X4_c= (df_X4_c - df_c.mean())
for i in df_X4_c.index:
    df_X4_c.X4.loc[i] = ((-np.delete(vector_c,3).dot(df_X4_c.loc[i].drop('X4')))/vector_c[3])
    df_X4_c.X4.loc[i] = round(df_X4_c.X4.loc[i] + df_c.mean()['X4'])
    df_X4_c.X1.loc[i]= df_X4_c.X1.loc[i] + df_c.mean()['X1']
    df_X4_c.X2.loc[i]= df_X4_c.X2.loc[i] + df_c.mean()['X2']
    df_X4_c.X4.loc[i]= df_X4_c.X3.loc[i] + df_c.mean()['X3']
    #print(df_X4_c.X4[i])

df_X4_d= df_X4_d[['X1','X2','X3','X4']]
df_X4_d= (df_X4_d - df_d.mean())
for i in df_X4_d.index:
    df_X4_d.X4.loc[i] = ((-np.delete(vector_d,3).dot(df_X4_d.loc[i].drop('X4')))/vector_d[3])
    df_X4_d.X4.loc[i] = round(df_X4_d.X4.loc[i] + df_d.mean()['X4'])
    df_X4_d.X1.loc[i]= df_X4_d.X1.loc[i] + df_d.mean()['X1']
    df_X4_d.X2.loc[i]= df_X4_d.X2.loc[i] + df_d.mean()['X2']
    df_X4_d.X4.loc[i]= df_X4_d.X3.loc[i] + df_d.mean()['X3']
    #print(df_X4_d.X4[i])
   
df_X4_e= df_X4_e[['X1','X2','X3','X4']]
df_X4_e= (df_X4_e - df_e.mean())
for i in df_X4_e.index:
    df_X4_e.X4.loc[i] = ((-np.delete(vector_e,3).dot(df_X4_e.loc[i].drop('X4')))/vector_e[3])
    df_X4_e.X4.loc[i] = round(df_X4_e.X4.loc[i] + df_e.mean()['X4'])
    df_X4_e.X1.loc[i]= df_X4_e.X1.loc[i] + df_e.mean()['X1']
    df_X4_e.X2.loc[i]= df_X4_e.X2.loc[i] + df_e.mean()['X2']
    df_X4_e.X4.loc[i]= df_X4_e.X3.loc[i] + df_e.mean()['X3']
    #print(df_X4_e.X4[i])


# In[329]:


#Filling missing Q1 values
for i in df_Q1.index:
    if df_Q1.X2[i] > df3[['X2']].mean()[0]:
        df_Q1.Q1.loc[i] = '1'
    else: 
        df_Q1.Q1.loc[i] = '0'
    #print(df_Q1.Q1[i])


# In[330]:


#Filling missing Q2 values
for i in df_Q2.index:
    if df_Q2.X3[i] > df3[['X3']].mean()[0]:
        df_Q2.Q2.loc[i] = '1'
    else: 
        df_Q2.Q2.loc[i] = '0'
    #print(df_Q2.Q2[i])


# In[331]:


#When Both X1 and X2 are unknown.
df_X1_X2 = df_NaN.loc[(df_NaN['X1'].isnull()) & (df_NaN['X2'].isnull())]
df_X1_X2_b= df_X1_X2.loc[(df_X1_X2['Q1']==0) & (df_X1_X2['Q2']==0)]
df_X1_X2_c= df_X1_X2.loc[(df_X1_X2['Q1']==0) & (df_X1_X2['Q2']==1)]
df_X1_X2_d= df_X1_X2.loc[(df_X1_X2['Q1']==1) & (df_X1_X2['Q2']==0)]
df_X1_X2_e= df_X1_X2.loc[(df_X1_X2['Q1']==1) & (df_X1_X2['Q2']==1)]

for i in df_X1_X2_b.index:
    df_X1_X2_b.X1.loc[i] = round(df_b.X1.mean())
    df_X1_X2_b.X2.loc[i] = round(df_b.X2.mean())
    #print(df_X1_X2_b.loc[i])

for i in df_X1_X2_c.index:
    df_X1_X2_c.X1.loc[i] = round(df_c.X1.mean())
    df_X1_X2_c.X2.loc[i] = round(df_c.X2.mean())
    #print(df_X1_X2_c.loc[i])
    
for i in df_X1_X2_d.index:
    df_X1_X2_d.X1.loc[i] = round(df_d.X1.mean())
    df_X1_X2_d.X2.loc[i] = round(df_d.X2.mean())
    #print(df_X1_X2_d.loc[i])
    
for i in df_X1_X2_e.index:
    df_X1_X2_e.X1.loc[i] = round(df_e.X1.mean())
    df_X1_X2_e.X2.loc[i] = round(df_e.X2.mean())
    #print(df_X1_X2_e.loc[i])


# In[332]:


#When Both X2 and X3 are unknown.
df_X2_X3 = df_NaN.loc[(df_NaN['X2'].isnull()) & (df_NaN['X3'].isnull())]
df_X2_X3_b= df_X2_X3.loc[(df_X2_X3['Q1']==0) & (df_X2_X3['Q2']==0)]
df_X2_X3_c= df_X2_X3.loc[(df_X2_X3['Q1']==0) & (df_X2_X3['Q2']==1)]
df_X2_X3_d= df_X2_X3.loc[(df_X2_X3['Q1']==1) & (df_X2_X3['Q2']==0)]
df_X2_X3_e= df_X2_X3.loc[(df_X2_X3['Q1']==1) & (df_X2_X3['Q2']==1)]

for i in df_X2_X3_b.index:
    df_X2_X3_b.X2.loc[i] = round(df_b.X2.mean())
    df_X2_X3_b.X3.loc[i] = round(df_b.X3.mean())
    #print(df_X2_X3_b.loc[i])

for i in df_X2_X3_c.index:
    df_X2_X3_c.X2.loc[i] = round(df_c.X2.mean())
    df_X2_X3_c.X3.loc[i] = round(df_c.X3.mean())
    #print(df_X2_X3_c.loc[i])
    
for i in df_X2_X3_d.index:
    df_X2_X3_d.X2.loc[i] = round(df_d.X2.mean())
    df_X2_X3_d.X3.loc[i] = round(df_d.X3.mean())
    #print(df_X2_X3_d.loc[i])
    
for i in df_X2_X3_e.index:
    df_X2_X3_e.X2.loc[i] = round(df_e.X2.mean())
    df_X2_X3_e.X3.loc[i] = round(df_e.X3.mean())
    #print(df_X2_X3_e.loc[i])


# In[333]:


#When Both X3 and X4 are unknown.
df_X3_X4 = df_NaN.loc[(df_NaN['X3'].isnull()) & (df_NaN['X4'].isnull())]
df_X3_X4_b= df_X3_X4.loc[(df_X3_X4['Q1']==0) & (df_X3_X4['Q2']==0)]
df_X3_X4_c= df_X3_X4.loc[(df_X3_X4['Q1']==0) & (df_X3_X4['Q2']==1)]
df_X3_X4_d= df_X3_X4.loc[(df_X3_X4['Q1']==1) & (df_X3_X4['Q2']==0)]
df_X3_X4_e= df_X3_X4.loc[(df_X3_X4['Q1']==1) & (df_X3_X4['Q2']==1)]

for i in df_X3_X4_b.index:
    df_X3_X4_b.X3.loc[i] = round(df_b.X3.mean())
    df_X3_X4_b.X4.loc[i] = round(df_b.X4.mean())
    #print(df_X3_X4_b.loc[i])

for i in df_X3_X4_c.index:
    df_X3_X4_c.X3.loc[i] = round(df_c.X3.mean())
    df_X3_X4_c.X4.loc[i] = round(df_c.X4.mean())
    #print(df_X3_X4_c.loc[i])
    
for i in df_X3_X4_d.index:
    df_X3_X4_d.X3.loc[i] = round(df_d.X3.mean())
    df_X3_X4_d.X4.loc[i] = round(df_d.X4.mean())
    #print(df_X3_X4_d.loc[i])
    
for i in df_X3_X4_e.index:
    df_X3_X4_e.X3.loc[i] = round(df_e.X3.mean())
    df_X3_X4_e.X4.loc[i] = round(df_e.X4.mean())
    #print(df_X3_X4_e.loc[i])


# In[334]:


df_X1_bb = df_X1.loc[(df_X1['Q1']==0) & (df_X1['Q2']==0)]
df_X1_bb.X1 = df_X1_b.X1 
df_X1_cc = df_X1.loc[(df_X1['Q1']==0) & (df_X1['Q2']==1)]
df_X1_cc.X1 = df_X1_c.X1
df_X1_dd = df_X1.loc[(df_X1['Q1']==1) & (df_X1['Q2']==0)]
df_X1_dd.X1 = df_X1_d.X1 
df_X1_ee = df_X1.loc[(df_X1['Q1']==1) & (df_X1['Q2']==1)]
df_X1_ee.X1 = df_X1_e.X1 

df_X2_bb = df_X2.loc[(df_X2['Q1']==0) & (df_X2['Q2']==0)]
df_X2_bb.X2 = df_X2_b.X2 
df_X2_cc = df_X2.loc[(df_X2['Q1']==0) & (df_X2['Q2']==1)]
df_X2_cc.X2 = df_X2_c.X2
df_X2_dd = df_X2.loc[(df_X2['Q1']==1) & (df_X2['Q2']==0)]
df_X2_dd.X2 = df_X2_d.X2 
df_X2_ee = df_X2.loc[(df_X2['Q1']==1) & (df_X2['Q2']==1)]
df_X2_ee.X2 = df_X2_e.X2 

df_X3_bb = df_X3.loc[(df_X3['Q1']==0) & (df_X3['Q2']==0)]
df_X3_bb.X3 = df_X3_b.X3
df_X3_cc = df_X3.loc[(df_X3['Q1']==0) & (df_X3['Q2']==1)]
df_X3_cc.X3 = df_X3_c.X3
df_X3_dd = df_X3.loc[(df_X3['Q1']==1) & (df_X3['Q2']==0)]
df_X3_dd.X3 = df_X3_d.X3 
df_X3_ee = df_X3.loc[(df_X3['Q1']==1) & (df_X3['Q2']==1)]
df_X3_ee.X3 = df_X3_e.X3 

df_X4_bb = df_X4.loc[(df_X4['Q1']==0) & (df_X4['Q2']==0)]
df_X4_bb.X4 = df_X4_b.X4
df_X4_cc = df_X4.loc[(df_X4['Q1']==0) & (df_X4['Q2']==1)]
df_X4_cc.X4 = df_X4_c.X4
df_X4_dd = df_X4.loc[(df_X4['Q1']==1) & (df_X4['Q2']==0)]
df_X4_dd.X4 = df_X4_d.X4 
df_X4_ee = df_X4.loc[(df_X4['Q1']==1) & (df_X4['Q2']==1)]
df_X4_ee.X4 = df_X4_e.X4 


df_X1_concat = pd.concat([df_X1_bb, df_X1_cc, df_X1_dd, df_X1_ee])
df_X2_concat = pd.concat([df_X2_bb, df_X2_cc, df_X2_dd, df_X2_ee])
df_X3_concat = pd.concat([df_X3_bb, df_X3_cc, df_X3_dd, df_X3_ee])
df_X4_concat = pd.concat([df_X4_bb, df_X4_cc, df_X4_dd, df_X4_ee])
df_X1_X2_concat = pd.concat([df_X1_X2_b, df_X1_X2_c, df_X1_X2_d, df_X1_X2_e])
df_X2_X3_concat = pd.concat([df_X2_X3_b, df_X2_X3_c, df_X2_X3_d, df_X2_X3_e])
df_X3_X4_concat = pd.concat([df_X3_X4_b, df_X3_X4_c, df_X3_X4_d, df_X3_X4_e])
df_concat= pd.concat([df_X1_concat,df_X2_concat,df_X3_concat,df_X4_concat,df_X1_X2_concat,df_X2_X3_concat,df_X3_X4_concat, df_Q1,df_Q2])


# In[335]:


df_Imputed= pd.concat([df3,df_concat])

for i in df_Imputed.index:
    df_original.loc[i] = df_Imputed.loc[i]
    
df_original

