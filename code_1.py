#!/usr/bin/env python
# coding: utf-8

# In[2]:


# required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[3]:


# input data reading and showing its head
columns = ['age', 'workclass', 'fnlwgt', 'education',
'education_num',
'marital_status', 'occupation', 'relationship', 'ethnicity',
'gender','capital_gain','capital_loss','hours_per_week','country_of_origin','income']
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',names=columns)
df.head()


# In[4]:


# number of categories in each categorical column (the most diverse one is country_of_origin)
for col in df.select_dtypes(exclude='number').columns:
    print(df[col].value_counts().size)


# In[5]:


df.info()


# In[6]:


# ordinal encoding over categorical columns (changes done on a copy of data)
df1 = df.copy()
enc1 = OrdinalEncoder()
for col in df1.select_dtypes(exclude='number').columns:
    df1[col] = enc1.fit_transform(df1[col].to_numpy().reshape(-1,1)) 
df1


# In[7]:


# features and targets are seperation
X = df1.drop('income',axis=1)
y = df1['income']
# test and train sets are formed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# data is scaled to be normalized without data-leakage
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train


# In[8]:


# neural net model is initialized
model1 = Sequential([Input(shape=(14,)),
                   Dense(20,activation='relu'),
                   Dense(5,activation='relu'),
                   Dense(1)])


# In[9]:


# model compiling and training
model1.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model1.fit(x=X_train, y=y_train, epochs=50, batch_size=32)

# loss values extraction
loss_df1 = pd.DataFrame(model1.history.history)


# In[15]:


loss_df1


# In[14]:


# plotting loss and accuracy over epochs
fig,ax = plt.subplots(1,2,figsize=[12,5])
ax[0].plot(loss_df1['loss'])
ax[0].set_title('loss over epochs')
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('Cross Entropy')
ax[0].grid()
ax[1].plot(loss_df1['accuracy'])
ax[1].set_title('accuracy over epochs')
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('Accuracy')
ax[1].grid()


# In[45]:


# random forest classification method
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_pred,y_test))


# In[ ]:




