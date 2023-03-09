#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df_all=pd.read_csv('diabetic_data_processed_withweight_short.csv',';')
print(type(df_all))
to_del = ['admission_source_id','encounter_id', 'patient_nbr','medical_specialty','payer_code','index','admission_type_id','discharge_disposition_id',
          'nateglinide','chlorpropamide','acetohexamide','tolbutamide','acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone']        
print (to_del)
#Filter_selected cols
filtered_cols = [c for c in df_all.columns if (c not in to_del) ]#and ('ENF' not in c)
df_2 = df_all[filtered_cols]
print ("df_2",df_2.shape)


# In[2]:


from sklearn.model_selection import train_test_split
import numpy as np
df_3 = df_2.fillna(value=np.mean(df_2,axis=0),inplace=False,axis=0).values
print ("df_3",df_3.shape)
print(df_2.columns)
X=df_3[:,0:-1]
y=df_3[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train=np.where(y_train > 0, 1, y_train)
y_test=np.where(y_test > 0, 1, y_test)
print(set(y_train))
print(set(y_test))
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[3]:


from sklearn.preprocessing import StandardScaler
# Define the scaler 
scaler = StandardScaler().fit(X_train)
# Scale the train set
X_train = scaler.transform(X_train)
# Scale the test set
X_test = scaler.transform(X_test)


# In[4]:


# Import `Sequential` from `keras.models`
from keras.models import Sequential
# Import `Dense` from `keras.layers`
from keras.layers import Dense
# Initialize the constructor
model = Sequential()
# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(27,)))
# Add one hidden layer 
model.add(Dense(8, activation='relu'))
# Add an output layer 
model.add(Dense(1, activation='sigmoid'))
# Model output shape
model.output_shape
# Model summary
model.summary()
# Model config
model.get_config()
# List all weight tensors 
model.get_weights()


# In[5]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])            
model.fit(X_train, y_train,epochs=2, batch_size=1, verbose=1)


# In[6]:


# Import the modules from `sklearn.metrics`
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
y_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test,verbose=1)
print(score)

# Confusion matrix
print(confusion_matrix(y_test, y_pred.round()))
       
# Precision 
print("Precision: ",precision_score(y_test, y_pred.round()))

# Recall
print("Recall: ",recall_score(y_test, y_pred.round()))

# F1 score
print("F1score: ", f1_score(y_test,y_pred.round()))


# In[7]:


import shap
explainer = shap.Explainer(model, np.array(X_train), feature_names=df_2.columns)
shap_values = explainer(X)


# In[8]:


shap.plots.bar(shap_values)


# In[9]:


shap.plots.bar(shap_values, max_display=28)

