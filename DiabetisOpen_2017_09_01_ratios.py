
# coding: utf-8

# In[2]:

from sklearn import preprocessing
from time import time
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_samples, silhouette_score
from operator import truediv
from sklearn.metrics import pairwise_distances
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

get_ipython().magic('matplotlib')

df=pd.read_csv('C:/diabetic_data_processed_withweight.csv',';')
to_del = ['encounter_id', 'patient_nbr','medical_specialty','payer_code','index','admission_type_id','discharge_disposition_id','admission_source_id']
print (to_del)
#Filter_selected cols
filtered_cols = [c for c in df.columns if (c not in to_del) ]#and ('ENF' not in c)
df_2 = df[filtered_cols]
print ("df_2",df_2.shape)
print(df_2.columns)

# Filter complete null columns
cols = np.where((np.sum(df_2.isnull(), axis=0).values) == df_2.shape[0])[0]
print (cols)
filt_cols = [c for c in df_2.columns if c not in df_2.columns[cols]]
df_3 = df_2[filt_cols]
print ("df_3",df_3.shape)

#Fill na
df_4 = df_3.fillna(value=np.mean(df_3,axis=0),inplace=False,axis=0).values
print ("df_4",df_4.shape)
data=df_4

selector = VarianceThreshold(threshold=(.99 * (1 - .99)))
newdata=selector.fit_transform(data)
idxs = selector.get_support(indices=True)
print(data[:, idxs])
print("indices",idxs)
columnslist=df_2.columns.tolist()
print("lenindex",len(idxs))
for z in range(0,len(columnslist)):
    if z not in idxs:
        print(columnslist[z])
print("after",newdata.shape)
print("initial",data.shape)
print("Headers_FINAL: ", df_2.columns.values.tolist())


# In[27]:

for i in range (0,df_2['diag_3'].size):
    if ((df_2.loc[i,'diag_1']==410 or df_2.loc[i,'diag_1']==412) or (df_2.loc[i,'diag_2']==410 or df_2.loc[i,'diag_2']==412) or (df_2.loc[i,'diag_3']==410 or df_2.loc[i,'diag_3']==412)):
        #print(df_2['diag_1'],df_2['diag_2'],df_2['diag_3'])
        df_2.loc[i,'Myocardial_infarction']=1
    else:
        df_2.loc[i,'Myocardial_infarction']=0


# In[41]:

df_2w = df_2[df_2['weight']>0]
df_2a = df_2[df_2['A1Cresult']>0]
print("df_2w: ", df_2w.shape)
print("df_2a: ", df_2a.shape)


# In[7]:

print(df_2['Myocardial_infarction'].astype('category').value_counts())
listA=df_2w[df_2w['Myocardial_infarction']==1]
listC=df_2w[df_2w['Myocardial_infarction']==0] 
print(df_2w['weight'].describe())
mean=df_2w['weight'].mean()
print("mean",mean)
listB=df_2w[df_2w['weight']>mean]
listD=df_2w[df_2w['weight']<mean]
#No healty and bad social
interA=listB[listB.isin(listA)]
interA=interA[interA['weight']>0]
a=len(interA)
#Healthy and bad social
interB=listB[listB.isin(listC)]
interB=interB[interB['weight']>0]
b=len(interB)
#No healty and social
interC=listD[listD.isin(listA)]
interC=interC[interC['weight']>0]
c=len(interC)
#Healthy and social
interD=listD[listD.isin(listC)]
interD=interD[interD['weight']>0]
d=len(interD)

print(len(listB), len(listA), len(listC),len(listD))
print(a,b,c,d)

OR=(a*d)/(c*d)
RR=(a/(a+b)) / (c/(c+d))
RD=(a/(a+b)) - (c/(c+d))
print("Weight-->","RR: ",RR,", RD: ",RD,", OR: ",OR)

print(df_2a['A1Cresult'].describe())
print("mean",mean)

listA=df_2a[df_2a['Myocardial_infarction']==1]
listC=df_2a[df_2a['Myocardial_infarction']==0] 
print(df_2a['A1Cresult'].describe())
mean=df_2a['A1Cresult'].mean()
print("mean",mean)
listB=df_2a[df_2a['A1Cresult']>6.5]
listD=df_2a[df_2a['A1Cresult']<6.5]
#No healty and bad social
interA=listB[listB.isin(listA)]
interA=interA[interA['A1Cresult']>0]
a=len(interA)
#Healthy and bad social
interB=listB[listB.isin(listC)]
interB=interB[interB['A1Cresult']>0]
b=len(interB)
#No healty and social
interC=listD[listD.isin(listA)]
interC=interC[interC['A1Cresult']>0]
c=len(interC)
#Healthy and social
interD=listD[listD.isin(listC)]
interD=interD[interD['A1Cresult']>0]
d=len(interD)

print(len(listB), len(listA), len(listC),len(listD))
print(a,b,c,d)

OR=(a*d)/(c*d)
RR=(a/(a+b)) / (c/(c+d))
RD=(a/(a+b)) - (c/(c+d))
print("A1Cresult-->","RR: ",RR,", RD: ",RD,", OR: ",OR)
#Characteristics
listMI=[]
for x in range (0,len(df_2)):
    if (df_2.loc[x,'Myocardial_infarction']==1):
        listMI.append(df_2.loc[x,:])    
df_MI = pd.DataFrame(listMI)
df_MI['age'].describe()

import scipy as sp
import scipy.stats
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h
print("Age",mean_confidence_interval(df_MI['age']))
print("Gender",df_MI['gender'].astype('category').value_counts())
print("Race",df_MI['race'].astype('category').value_counts())
print("Readmitted",df_MI['readmitted'].astype('category').value_counts())


# In[28]:

for i in range (0,df_2['diag_3'].size):
    if (df_2.loc[i,'diag_1']==428 or df_2.loc[i,'diag_2']==428 or (df_2.loc[i,'diag_3']==428)):
        #print(df_2['diag_1'],df_2['diag_2'],df_2['diag_3'])
        df_2.loc[i,'Congestive_heart_failure']=1
    else:
        df_2.loc[i,'Congestive_heart_failure']=0      


# In[55]:

df_2w = df_2[df_2['weight']>0]
df_2a = df_2[df_2['A1Cresult']>0]
print("df_2w: ", df_2w.shape)
print("df_2a: ", df_2a.shape)


# In[56]:

print(df_2['Congestive_heart_failure'].astype('category').value_counts())
listA=df_2w[df_2w['Congestive_heart_failure']==1]
listC=df_2w[df_2w['Congestive_heart_failure']==0] 
print(df_2w['weight'].describe())
mean=df_2w['weight'].mean()
print("mean",mean)
listB=df_2w[df_2w['weight']>mean]
listD=df_2w[df_2w['weight']<mean]
#No healty and bad social
interA=listB[listB.isin(listA)]
interA=interA[interA['weight']>0]
a=len(interA)
#Healthy and bad social
interB=listB[listB.isin(listC)]
interB=interB[interB['weight']>0]
b=len(interB)
#No healty and social
interC=listD[listD.isin(listA)]
interC=interC[interC['weight']>0]
c=len(interC)
#Healthy and social
interD=listD[listD.isin(listC)]
interD=interD[interD['weight']>0]
d=len(interD)

print(len(listB), len(listA), len(listC),len(listD))
print(a,b,c,d)

OR=(a*d)/(c*d)
RR=(a/(a+b)) / (c/(c+d))
RD=(a/(a+b)) - (c/(c+d))
print("Weight-->","RR: ",RR,", RD: ",RD,", OR: ",OR)

listA=df_2a[df_2a['Congestive_heart_failure']==1]
listC=df_2a[df_2a['Congestive_heart_failure']==0] 
print(df_2a['A1Cresult'].describe())
mean=df_2a['A1Cresult'].mean()
print("mean",mean)
listB=df_2a[df_2a['A1Cresult']>6.5]
listD=df_2a[df_2a['A1Cresult']<6.5]
#No healty and bad social
interA=listB[listB.isin(listA)]
interA=interA[interA['A1Cresult']>0]
a=len(interA)
#Healthy and bad social
interB=listB[listB.isin(listC)]
interB=interB[interB['A1Cresult']>0]
b=len(interB)
#No healty and social
interC=listD[listD.isin(listA)]
interC=interC[interC['A1Cresult']>0]
c=len(interC)
#Healthy and social
interD=listD[listD.isin(listC)]
interD=interD[interD['A1Cresult']>0]
d=len(interD)

print(len(listB), len(listA), len(listC),len(listD))
print(a,b,c,d)

OR=(a*d)/(c*d)
RR=(a/(a+b)) / (c/(c+d))
RD=(a/(a+b)) - (c/(c+d))
print("A1CResult-->","RR: ",RR,", RD: ",RD,", OR: ",OR)
#Characteristics
listCHF=[]
for x in range (0,len(df_2)):
    if (df_2.loc[x,'Congestive_heart_failure']==1):
        listCHF.append(df_2.loc[x,:])    
df_CHF = pd.DataFrame(listCHF)
df_CHF['age'].describe()

print("Age",mean_confidence_interval(df_CHF['age']))
print("Gender",df_CHF['gender'].astype('category').value_counts())
print("Race",df_CHF['race'].astype('category').value_counts())
print("Readmitted",df_CHF['readmitted'].astype('category').value_counts())


# In[29]:

for i in range (0,df_2['diag_3'].size):
    if (df_2.loc[i,'diag_1'] in (443,441,785,43) 
        or df_2.loc[i,'diag_2'] in (443,441,785,43)  
        or df_2.loc[i,'diag_3'] in (443,441,785,43)):
        df_2.loc[i,'Peripheral_vascular_disease']=1
    else:
        df_2.loc[i,'Peripheral_vascular_disease']=0


# In[58]:

df_2w = df_2[df_2['weight']>0]
df_2a = df_2[df_2['A1Cresult']>0]
print("df_2w: ", df_2w.shape)
print("df_2a: ", df_2a.shape)


# In[59]:

print(df_2['Peripheral_vascular_disease'].astype('category').value_counts())
listA=df_2w[df_2w['Peripheral_vascular_disease']==1]
listC=df_2w[df_2w['Peripheral_vascular_disease']==0] 
print(df_2w['weight'].describe())
mean=df_2w['weight'].mean()
print("mean",mean)
listB=df_2w[df_2w['weight']>mean]
listD=df_2w[df_2w['weight']<mean]
#No healty and bad social
interA=listB[listB.isin(listA)]
interA=interA[interA['weight']>0]
a=len(interA)
#Healthy and bad social
interB=listB[listB.isin(listC)]
interB=interB[interB['weight']>0]
b=len(interB)
#No healty and social
interC=listD[listD.isin(listA)]
interC=interC[interC['weight']>0]
c=len(interC)
#Healthy and social
interD=listD[listD.isin(listC)]
interD=interD[interD['weight']>0]
d=len(interD)

print(len(listB), len(listA), len(listC),len(listD))
print(a,b,c,d)

OR=(a*d)/(c*d)
RR=(a/(a+b)) / (c/(c+d))
RD=(a/(a+b)) - (c/(c+d))
print("Weight-->","RR: ",RR,", RD: ",RD,", OR: ",OR)

listA=df_2a[df_2a['Peripheral_vascular_disease']==1]
listC=df_2a[df_2a['Peripheral_vascular_disease']==0] 
print(df_2a['A1Cresult'].describe())
mean=df_2a['A1Cresult'].mean()
print("mean",mean)
listB=df_2a[df_2a['A1Cresult']>6.5]
listD=df_2a[df_2a['A1Cresult']<6.5]
#No healty and bad social
interA=listB[listB.isin(listA)]
interA=interA[interA['A1Cresult']>0]
a=len(interA)
#Healthy and bad social
interB=listB[listB.isin(listC)]
interB=interB[interB['A1Cresult']>0]
b=len(interB)
#No healty and social
interC=listD[listD.isin(listA)]
interC=interC[interC['A1Cresult']>0]
c=len(interC)
#Healthy and social
interD=listD[listD.isin(listC)]
interD=interD[interD['A1Cresult']>0]
d=len(interD)

print(len(listB), len(listA), len(listC),len(listD))
print(a,b,c,d)

OR=(a*d)/(c*d)
RR=(a/(a+b)) / (c/(c+d))
RD=(a/(a+b)) - (c/(c+d))
print("A1CResult-->","RR: ",RR,", RD: ",RD,", OR: ",OR)
#Characteristics
listCHF=[]
for x in range (0,len(df_2)):
    if (df_2.loc[x,'Peripheral_vascular_disease']==1):
        listCHF.append(df_2.loc[x,:])    
df_CHF = pd.DataFrame(listCHF)
print(df_CHF['age'].describe())

print("Age",mean_confidence_interval(df_CHF['age']))
print("Gender",df_CHF['gender'].astype('category').value_counts())
print("Race",df_CHF['race'].astype('category').value_counts())
print("Readmitted",df_CHF['readmitted'].astype('category').value_counts())


# In[21]:

import scipy.stats as stats
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


# In[3]:

for i in range (0,df_2['diag_3'].size):
    if ((df_2.loc[i,'diag_1']>=430 and df_2.loc[i,'diag_1']<=438) or (df_2.loc[i,'diag_2']>=430 and df_2.loc[i,'diag_2']<=438) or (df_2.loc[i,'diag_3']>=430 and df_2.loc[i,'diag_3']<=438)):
        df_2.loc[i,'Cerebrovascular_disease']=1
    else:
        df_2.loc[i,'Cerebrovascular_disease']=0


# In[4]:

df_2w = df_2[df_2['weight']>0]
df_2a = df_2[df_2['A1Cresult']>0]
print("df_2w: ", df_2w.shape)
print("df_2a: ", df_2a.shape)


# In[30]:

print(df_2['Cerebrovascular_disease'].astype('category').value_counts())
listA=df_2w[df_2w['Cerebrovascular_disease']==1]
listC=df_2w[df_2w['Cerebrovascular_disease']==0] 
print(df_2w['weight'].describe())
mean=df_2w['weight'].mean()
print("mean",mean)
listB=df_2w[df_2w['weight']>mean]
listD=df_2w[df_2w['weight']<mean]
#No healty and bad social
interA=listB[listB.isin(listA)]
interA=interA[interA['weight']>0]
a=len(interA)
#Healthy and bad social
interB=listB[listB.isin(listC)]
interB=interB[interB['weight']>0]
b=len(interB)
#No healty and social
interC=listD[listD.isin(listA)]
interC=interC[interC['weight']>0]
c=len(interC)
#Healthy and social
interD=listD[listD.isin(listC)]
interD=interD[interD['weight']>0]
d=len(interD)

print(len(listB), len(listA), len(listC),len(listD))
print(a,b,c,d)

OR=(a*d)/(c*d)
RR=(a/(a+b)) / (c/(c+d))
RD=(a/(a+b)) - (c/(c+d))
print("Weight-->","RR: ",RR,", RD: ",RD,", OR: ",OR)

listA=df_2a[df_2a['Cerebrovascular_disease']==1]
listC=df_2a[df_2a['Cerebrovascular_disease']==0] 
print(df_2a['A1Cresult'].describe())
mean=df_2a['A1Cresult'].mean()
print("mean",mean)
listB=df_2a[df_2a['A1Cresult']>6.5]
listD=df_2a[df_2a['A1Cresult']<6.5]
#No healty and bad social
interA=listB[listB.isin(listA)]
interA=interA[interA['A1Cresult']>0]
a=len(interA)
#Healthy and bad social
interB=listB[listB.isin(listC)]
interB=interB[interB['A1Cresult']>0]
b=len(interB)
#No healty and social
interC=listD[listD.isin(listA)]
interC=interC[interC['A1Cresult']>0]
c=len(interC)
#Healthy and social
interD=listD[listD.isin(listC)]
interD=interD[interD['A1Cresult']>0]
d=len(interD)

print(len(listB), len(listA), len(listC),len(listD))
print(a,b,c,d)

OR=(a*d)/(c*d)
RR=(a/(a+b)) / (c/(c+d))
RD=(a/(a+b)) - (c/(c+d))
print("A1CResult-->","RR: ",RR,", RD: ",RD,", OR: ",OR)
#Characteristics
listCHF=[]
for x in range (0,len(df_2)):
    if (df_2.loc[x,'Cerebrovascular_disease']==1):
        listCHF.append(df_2.loc[x,:])    
df_CHF = pd.DataFrame(listCHF)
print(df_CHF['age'].describe())

print("Age",mean_confidence_interval(df_CHF['age']))
print("Gender",df_CHF['gender'].astype('category').value_counts())
print("Race",df_CHF['race'].astype('category').value_counts())
print("Readmitted",df_CHF['readmitted'].astype('category').value_counts())


# In[14]:

for i in range (0,df_2['diag_3'].size):
    if (df_2.loc[i,'diag_1']==290 or df_2.loc[i,'diag_2']==290 or df_2.loc[i,'diag_3']==290):
        df_2.loc[i,'Dementia']=1
    else:
        df_2.loc[i,'Dementia']=0


# In[17]:

df_2w = df_2[df_2['weight']>0]
df_2a = df_2[df_2['A1Cresult']>0]
print("df_2w: ", df_2w.shape)
print("df_2a: ", df_2a.shape)


# In[19]:

import scipy.stats as stats


# In[22]:

print(df_2['Dementia'].astype('category').value_counts())
listA=df_2w[df_2w['Dementia']==1]
listC=df_2w[df_2w['Dementia']==0] 
print(df_2w['weight'].describe())
mean=df_2w['weight'].mean()
print("mean",mean)
listB=df_2w[df_2w['weight']>mean]
listD=df_2w[df_2w['weight']<mean]
#No healty and bad social
interA=listB[listB.isin(listA)]
interA=interA[interA['weight']>0]
a=len(interA)
#Healthy and bad social
interB=listB[listB.isin(listC)]
interB=interB[interB['weight']>0]
b=len(interB)
#No healty and social
interC=listD[listD.isin(listA)]
interC=interC[interC['weight']>0]
c=len(interC)
#Healthy and social
interD=listD[listD.isin(listC)]
interD=interD[interD['weight']>0]
d=len(interD)

print(len(listB), len(listA), len(listC),len(listD))
print(a,b,c,d)

OR=(a*d)/(c*d)
#RR=(a/(a+b)) / (c/(c+d))
#RD=(a/(a+b)) - (c/(c+d))
print("Weight-->","RR: ",RR,", RD: ",RD,", OR: ",OR)

listA=df_2a[df_2a['Dementia']==1]
listC=df_2a[df_2a['Dementia']==0] 
print(df_2a['A1Cresult'].describe())
mean=df_2a['A1Cresult'].mean()
print("mean",mean)
listB=df_2a[df_2a['A1Cresult']>6.5]
listD=df_2a[df_2a['A1Cresult']<6.5]
#No healty and bad social
interA=listB[listB.isin(listA)]
interA=interA[interA['A1Cresult']>0]
a=len(interA)
#Healthy and bad social
interB=listB[listB.isin(listC)]
interB=interB[interB['A1Cresult']>0]
b=len(interB)
#No healty and social
interC=listD[listD.isin(listA)]
interC=interC[interC['A1Cresult']>0]
c=len(interC)
#Healthy and social
interD=listD[listD.isin(listC)]
interD=interD[interD['A1Cresult']>0]
d=len(interD)

print(len(listB), len(listA), len(listC),len(listD))
print(a,b,c,d)

OR=(a*d)/(c*d)
RR=(a/(a+b)) / (c/(c+d))
RD=(a/(a+b)) - (c/(c+d))
print("A1CResult-->","RR: ",RR,", RD: ",RD,", OR: ",OR)
#Characteristics
listCHF=[]
for x in range (0,len(df_2)):
    if (df_2.loc[x,'Dementia']==1):
        listCHF.append(df_2.loc[x,:])    
df_CHF = pd.DataFrame(listCHF)
print(df_CHF['age'].describe())

print("Age",mean_confidence_interval(df_CHF['age']))
print("Gender",df_CHF['gender'].astype('category').value_counts())
print("Race",df_CHF['race'].astype('category').value_counts())
print("Readmitted",df_CHF['readmitted'].astype('category').value_counts())


# In[23]:

for i in range (0,df_2['diag_3'].size):
    if ((df_2.loc[i,'diag_1']>=490 and df_2.loc[i,'diag_1']<=506) or (df_2.loc[i,'diag_2']>=490 and df_2.loc[i,'diag_2']<=506) or (df_2.loc[i,'diag_3']>=490 and df_2.loc[i,'diag_3']<=506)):
        df_2.loc[i,'Chronic_pulmonary_disease']=1
    else:
        df_2.loc[i,'Chronic_pulmonary_disease']=0


# In[25]:

df_2w = df_2[df_2['weight']>0]
df_2a = df_2[df_2['A1Cresult']>0]
print("df_2w: ", df_2w.shape)
print("df_2a: ", df_2a.shape)


# In[26]:

print(df_2['Chronic_pulmonary_disease'].astype('category').value_counts())
listA=df_2w[df_2w['Chronic_pulmonary_disease']==1]
listC=df_2w[df_2w['Chronic_pulmonary_disease']==0] 
print(df_2w['weight'].describe())
mean=df_2w['weight'].mean()
print("mean",mean)
listB=df_2w[df_2w['weight']>mean]
listD=df_2w[df_2w['weight']<mean]
#No healty and bad social
interA=listB[listB.isin(listA)]
interA=interA[interA['weight']>0]
a=len(interA)
#Healthy and bad social
interB=listB[listB.isin(listC)]
interB=interB[interB['weight']>0]
b=len(interB)
#No healty and social
interC=listD[listD.isin(listA)]
interC=interC[interC['weight']>0]
c=len(interC)
#Healthy and social
interD=listD[listD.isin(listC)]
interD=interD[interD['weight']>0]
d=len(interD)

print(len(listB), len(listA), len(listC),len(listD))
print(a,b,c,d)

OR=(a*d)/(c*d)
RR=(a/(a+b)) / (c/(c+d))
RD=(a/(a+b)) - (c/(c+d))
print("Weight-->","RR: ",RR,", RD: ",RD,", OR: ",OR)

listA=df_2a[df_2a['Chronic_pulmonary_disease']==1]
listC=df_2a[df_2a['Chronic_pulmonary_disease']==0] 
print(df_2a['A1Cresult'].describe())
mean=df_2a['A1Cresult'].mean()
print("mean",mean)
listB=df_2a[df_2a['A1Cresult']>6.5]
listD=df_2a[df_2a['A1Cresult']<6.5]
#No healty and bad social
interA=listB[listB.isin(listA)]
interA=interA[interA['A1Cresult']>0]
a=len(interA)
#Healthy and bad social
interB=listB[listB.isin(listC)]
interB=interB[interB['A1Cresult']>0]
b=len(interB)
#No healty and social
interC=listD[listD.isin(listA)]
interC=interC[interC['A1Cresult']>0]
c=len(interC)
#Healthy and social
interD=listD[listD.isin(listC)]
interD=interD[interD['A1Cresult']>0]
d=len(interD)

print(len(listB), len(listA), len(listC),len(listD))
print(a,b,c,d)

OR=(a*d)/(c*d)
RR=(a/(a+b)) / (c/(c+d))
RD=(a/(a+b)) - (c/(c+d))
print("A1CResult-->","RR: ",RR,", RD: ",RD,", OR: ",OR)
#Characteristics
listCHF=[]
for x in range (0,len(df_2)):
    if (df_2.loc[x,'Chronic_pulmonary_disease']==1):
        listCHF.append(df_2.loc[x,:])    
df_CHF = pd.DataFrame(listCHF)
print(df_CHF['age'].describe())

print("Age",mean_confidence_interval(df_CHF['age']))
print("Gender",df_CHF['gender'].astype('category').value_counts())
print("Race",df_CHF['race'].astype('category').value_counts())
print("Readmitted",df_CHF['readmitted'].astype('category').value_counts())


# In[31]:

print(df_2.columns)


# In[32]:

to_del2 = ['diag_1', 'diag_2','diag_3','admission_type_id','discharge_disposition_id','admission_source_id',
           'nateglinide','chlorpropamide','acetohexamide','tolbutamide','acarbose','miglitol','troglitazone',
'tolazamide','examide','citoglipton','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone',
'metformin-rosiglitazone','metformin-pioglitazone']
print (to_del2)
#Filter_selected cols
filtered_cols = [c for c in df_2.columns if (c not in to_del2) ]#and ('ENF' not in c)
df_3 = df_2[filtered_cols]
print ("df_3",df_3.shape)
print(df_3.columns)


# In[62]:

mid = df_3['readmitted']
df_3.drop(labels=['readmitted'], axis=1,inplace = True)
df_3.insert(0, 'readmitted', mid)
print(df_3.shape)


msk = np.random.rand(len(df_3)) < 0.8
df, df_test = df_3[msk].copy(deep = True), df_3[~msk].copy(deep = True)
df = df.reset_index()
df_test = df_test.reset_index()
print("df.shape",df.shape)
print("df_test.shape",df_test.shape)
y_train=df['readmitted']
y_test=df_test['readmitted']
print(set(y_train))
print(set(y_test))

x_train=df.iloc[:,1:50]
x_test=df_test.iloc[:,1:50]
print(set(x_train))
print(set(x_test))


# In[34]:

print(df_3.columns)


# In[49]:

#Fill na
df_4 = df_3.fillna(value=np.mean(df_3,axis=0),inplace=False,axis=0).values
print ("df_4",df_4.shape)

scaler = preprocessing.MinMaxScaler().fit(df_4)
data = scaler.transform(df_4)
print ("data",data.shape)


# In[50]:

thrd = 0.8
total = 0
pca = PCA().fit(data)
reduced_data = pca.transform(data)
for pca_comps,r in enumerate(pca.explained_variance_ratio_):
    if total > thrd:
        break
    total += r
print ("Num pca_comps per >", thrd,"ratio:", pca_comps, total)
print ("Explained variance first 2 components",pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1])
print (pca.n_components_)
print (np.sum(pca.explained_variance_ratio_[:2]))

print ("PCA+K-means:", pca_comps)
print("1st component: ", pca.components_[0])
print("2nd component: ", pca.components_[1])


# In[51]:

from sklearn.metrics import silhouette_samples, silhouette_score
Resultk=[0]*9
ResultC=[0]*9
for k in [2,3,4,5,6,7,8,9,10]:
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
    cluster_labels = kmeans.fit_predict(reduced_data[:,:pca_comps])
    #cluster_labels=kmeans.fit(reduced_data[:,:2])
    #silhouette_avg = silhouette_score(reduced_data[:,:pca_comps], cluster_labels)
    #silhouette_avg = silhouette_score(reduced_data[:,:2], cluster_labels)
    #print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
    calinski_harabaz_score_avg = metrics.calinski_harabaz_score(reduced_data[:,:pca_comps], cluster_labels)
    #calinski_harabaz_score_avg = metrics.calinski_harabaz_score(reduced_data[:,:2], cluster_labels)
    print("For n_clusters =", k," the average metrics.calinski_harabaz_score is :", calinski_harabaz_score_avg)
    Resultk[k-2]=k
    ResultC[k-2]=calinski_harabaz_score_avg    
plt.plot(Resultk,ResultC,'r*-.')


# In[56]:

n_clusters = 2
kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
pca = PCA().fit(data)
cluster_labels = kmeans.fit_predict(reduced_data[:,:2])
plt.figure()
plt.plot(range(len(pca.explained_variance_ratio_)), np.cumsum(pca.explained_variance_ratio_))
plt.axvline(pca_comps, color="red")  
plt.ylim(0.0,1.1)
plt.show()


# In[57]:

n_clusters=2
reduced_data = pca.transform(data)
#print ("Reduced data: ",reduced_data.shape)
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
kmeans.fit(reduced_data[:,:2])

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min()-.05, reduced_data[:, 0].max()+.05
y_min, y_max = reduced_data[:, 1].min()-.05, reduced_data[:, 1].max()+.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(10,10))
#plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap= plt.cm.Pastel2,#cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

#plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=4)
print(df_4.shape)
print(data.shape)
print(reduced_data.shape)
"""
listA=df_3[df_3['readmitted'] == 0].index
plt.plot(reduced_data[listA, 0], reduced_data[listA, 1],'k.', markersize=4, c='g', label='No Readmitted')
listB=df_3[df_3['readmitted'] == 29].index
plt.plot(reduced_data[listB, 0],reduced_data[listB, 1],'k.', markersize=4, c='r',label ='<30')
listC=df_3[df_3['readmitted'] == 31].index
plt.plot(reduced_data[listC, 0],reduced_data[listC, 1],'k.', markersize=4, c='b',label ='>30')
"""
listA=df[df['change'] == 0].index
plt.plot(reduced_data[listA, 0], reduced_data[listA, 1],'k.', markersize=4, c='g', label='No disease')
listB=df[df['change'] == 1].index
plt.plot(reduced_data[listB, 0],reduced_data[listB, 1],'k.', markersize=4, c='r',label ='Disease')

#Peripheral_vascular_disease', 'Cerebrovascular_disease', 'Dementia',
#       'Chronic_pulmonary_disease', 'Myocardial_infarction',
#       'Congestive_heart_failure'

# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[0, 0], centroids[0, 1],
            marker='>', s=169, linewidths=3, 
            color='black', zorder=10)
plt.scatter(centroids[1, 0], centroids[1, 1],
            marker='H', s=169, linewidths=3, 
            color='black', zorder=10)
"""
plt.scatter(centroids[2, 0], centroids[2, 1],
            marker='>', s=169, linewidths=3, 
            color='blue', zorder=10)
plt.scatter(centroids[3, 0], centroids[3, 1],
            marker='H', s=169, linewidths=3, 
            color='blue', zorder=10)
plt.scatter(centroids[4, 0], centroids[4, 1],
            marker='>', s=169, linewidths=3, 
            color='purple', zorder=10)
plt.scatter(centroids[5, 0], centroids[5, 1],
            marker='H', s=169, linewidths=3, 
            color='purple', zorder=10)
            """
plt.title('K-means clustering on the diabetes dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.legend()
plt.show()


# In[58]:

n_clusters=2
print(set(cluster_labels))
num_Myocardial_infarction = [0] * n_clusters
for i in range(len(reduced_data[:,:2])):
    if (data[i][df_3.columns.get_loc("Myocardial_infarction")] == 0):
        num_Myocardial_infarction[cluster_labels[i]] = num_Myocardial_infarction[cluster_labels[i]] + 1
print ("cluster_sizes = ", [ len([1 for label in cluster_labels if label == i]) for i in range(n_clusters) ])
print ("Myocardial infarction = ", num_Myocardial_infarction)

num_Peripheral_vascular_disease = [0] * n_clusters
for i in range(len(reduced_data[:,:2])):
    if (data[i][df_3.columns.get_loc("Peripheral_vascular_disease")] == 0):
        num_Peripheral_vascular_disease[cluster_labels[i]] = num_Peripheral_vascular_disease[cluster_labels[i]] + 1
print ("cluster_sizes = ", [ len([1 for label in cluster_labels if label == i]) for i in range(n_clusters) ])
print ("num_no_Peripheral_vascular_disease = ", num_Peripheral_vascular_disease)

num_Cerebrovascular_disease = [0] * n_clusters
for i in range(len(reduced_data[:,:2])):
    if (data[i][df_3.columns.get_loc("Cerebrovascular_disease")] == 0):
        num_Cerebrovascular_disease[cluster_labels[i]] = num_Cerebrovascular_disease[cluster_labels[i]] + 1
print ("cluster_sizes = ", [ len([1 for label in cluster_labels if label == i]) for i in range(n_clusters) ])
print ("num_no_Cerebrovascular_disease = ", num_Cerebrovascular_disease)

num_Dementia = [0] * n_clusters
for i in range(len(reduced_data[:,:2])):
    if (data[i][df_3.columns.get_loc("Dementia")] == 0):
        num_Dementia[cluster_labels[i]] = num_Dementia[cluster_labels[i]] + 1
print ("cluster_sizes = ", [ len([1 for label in cluster_labels if label == i]) for i in range(n_clusters) ])
print ("num_no_Dementia = ", num_Dementia)

num_Chronic_pulmonary_disease = [0] * n_clusters
for i in range(len(reduced_data[:,:2])):
    if (data[i][df_3.columns.get_loc("Chronic_pulmonary_disease")] == 0):
        num_Chronic_pulmonary_disease[cluster_labels[i]] = num_Chronic_pulmonary_disease[cluster_labels[i]] + 1
print ("cluster_sizes = ", [ len([1 for label in cluster_labels if label == i]) for i in range(n_clusters) ])
print ("num_no_Chronic_pulmonary_disease = ", num_Chronic_pulmonary_disease)

num_Congestive_heart_failure = [0] * n_clusters
for i in range(len(reduced_data[:,:2])):
    if (data[i][df_3.columns.get_loc("Congestive_heart_failure")] == 0):
        num_Congestive_heart_failure[cluster_labels[i]] = num_Congestive_heart_failure[cluster_labels[i]] + 1
print ("cluster_sizes = ", [ len([1 for label in cluster_labels if label == i]) for i in range(n_clusters) ])
print ("num_no_Congestive_heart_failure = ", num_Congestive_heart_failure)

num_readmitted = [0] * n_clusters
for i in range(len(reduced_data[:,:2])):
    if (data[i][df_3.columns.get_loc("readmitted")] == 0):
        num_readmitted[cluster_labels[i]] = num_readmitted[cluster_labels[i]] + 1
print ("cluster_sizes = ", [ len([1 for label in cluster_labels if label == i]) for i in range(n_clusters) ])
print ("num_no_readmitted = ", num_readmitted)

num_age = [0] * n_clusters
for i in range(len(reduced_data[:,:2])):
        num_age[cluster_labels[i]] = num_age[cluster_labels[i]] + df_4[i][df_3.columns.get_loc("age")]
mida_cluster=[ len([1 for label in cluster_labels if label == i]) for i in range(n_clusters) ]
print ("average age = ", list(map(truediv,num_age,mida_cluster)))

num_age = [0] * n_clusters
for i in range(len(reduced_data[:,:2])):
        num_age[cluster_labels[i]] = num_age[cluster_labels[i]] + df_4[i][df_3.columns.get_loc("weight")]
mida_cluster=[ len([1 for label in cluster_labels if label == i]) for i in range(n_clusters) ]
print ("average weight = ", list(map(truediv,num_age,mida_cluster)))

num_age = [0] * n_clusters
for i in range(len(reduced_data[:,:2])):
        num_age[cluster_labels[i]] = num_age[cluster_labels[i]] + df_4[i][df_3.columns.get_loc("A1Cresult")]
mida_cluster=[ len([1 for label in cluster_labels if label == i]) for i in range(n_clusters) ]
print ("average A1Cresult = ", list(map(truediv,num_age,mida_cluster)))


# In[59]:

#P-values of attributes of each cluster
import scipy.stats as stats
#true_mu = 0
listaA=df_3.columns.tolist()
cluster1=df_4[kmeans.labels_==0]
cluster2=df_4[kmeans.labels_==1]
cluster_labels=kmeans.labels_
#print(cluster1.shape)
#print(cluster2.shape)
contador=0
for z in range(0,len(df_3.columns.tolist())):
    two_sample = stats.ttest_ind(cluster1[:,z],cluster2[:,z])
    if(two_sample[1]<0.05):contador=contador+1

w, h = 3, contador
Matrix = [[0.0 for x in range(w)] for y in range(h)] 
i=0
for z in range(0,len(df_3.columns.tolist())):
    #print(df_2.loc[1,z])
    #print(np.std(cluster1[:,z]))
    #print(df_2.values[:,z])
    two_sample = stats.ttest_ind(cluster1[:,z],cluster2[:,z])
    #two_sample = stats.chisquare(cluster1[:,8],cluster2[:,8])
    if(two_sample[1]<0.05): 
        #print(i)
        Matrix[i][0]=listaA[z]
        Matrix[i][1]='{0:.400f}'.format(two_sample[1])
        Matrix[i][2]=two_sample[1]
        i=i+1
arr = np.array(Matrix)
arr = arr[arr[:,1].argsort()]
#print(arr)

for z in range(0,h):
    if (0<1):
        num_hipertensos = [0] * n_clusters
        for i in range(len(data[:,:2])):
            num_hipertensos[cluster_labels[i]] = num_hipertensos[cluster_labels[i]] + df_4[i][df_3.columns.get_loc(arr[z,0])]
        mida_cluster=[ len([1 for label in cluster_labels if label == i]) for i in range(n_clusters) ]
        print ((arr[z,0]),": ", list(map(truediv,num_hipertensos,mida_cluster)),", p-value: ",arr[z,2])
    else:
        num_hipertensos = [0] * n_clusters
        for i in range(len(data[:,:2])):
            #print("z",z)
            #print("i",i)
            if (data[i][df_3.columns.get_loc(arr[z,0])] == 1):
                num_hipertensos[cluster_labels[i]] = num_hipertensos[cluster_labels[i]] + 1
        mida_cluster=[ len([1 for label in cluster_labels if label == i]) for i in range(n_clusters) ]
        percentage=list(map(truediv,num_hipertensos,mida_cluster))
        percentage[0]=round(percentage[0],2)
        percentage[1]=round(percentage[1],2)
        print ((arr[z,0]),": ", num_hipertensos,", percentage: ",percentage, "p-value: ",arr[z,2])


# In[65]:

#Decision tree to predict readmissions
print(newdata_test.shape)
print(newdata.shape)


y_train=newdata[:,0]
print("ytrain: ",y_train.shape)
x_train=newdata[:,1:]
print("xtrain: ",x_train.shape)
y_test=newdata_test[:,0]
print("y_test: ",y_test.shape)
x_test=newdata_test[:,1:]
print("x_test: ",x_test.shape)

y_train_encoded= np.array(["%.2f" % w for w in y_train.reshape(y_train.size)])
y_train_encoded = y_train_encoded.reshape(y_train.shape)
y_train_encoded=y_train_encoded.tolist()
#print("ytrain_encoded", y_train_encoded)
print(set(y_train_encoded))
y_test_encoded= np.array(["%.2f" % w for w in y_test.reshape(y_test.size)])
y_test_encoded = y_test_encoded.reshape(y_test.shape)
y_test_encoded=y_test_encoded.tolist()
#print("ytrain_encoded", y_train_encoded)
print(set(y_test_encoded))


from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from sklearn import tree
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score

#param_grid = {"criterion": ["gini", "entropy"],
#              "min_samples_split": [2, 10, 20],
#              "max_depth": [None, 2, 5, 10],
#              "min_samples_leaf": [1, 5, 10],
#              "max_leaf_nodes": [None, 5, 10, 20],
#              }

param_grid = {"criterion": ["gini"],
              "min_samples_split": [2],
              "max_depth": [None],
              "min_samples_leaf": [5],
              "max_leaf_nodes": [5],
              }

tree2 = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid)

tree2.fit(x_train, y_train_encoded)
y_pred = tree2.predict(x_test)

#max_depth=4
#clf = tree.DecisionTreeClassifier(random_state=13)
#X_train, X_test, y_train, y_test = train_test_split(data, kmeans.labels_, test_size=0.2)
#y_pred = clf.fit(df_4, y_train).predict(df_4_test)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_encoded, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
class_names=['0','<30','>30']
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
from sklearn.metrics import accuracy_score
print("Accuracy score", accuracy_score(y_test_encoded, y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test_encoded, y_pred, target_names=class_names))

print("Best parameters set found on development set:")
print()
print(tree2.best_params_)
    
with open("C://Users/laia.subirats/Documents/output_diabetes.dot", "w") as output_file:
    tree.export_graphviz(tree2.best_estimator_, out_file=output_file, feature_names=df_3.columns.tolist(),class_names=class_names)


# In[66]:

#SVM
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Set the parameters by cross-validation
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                     'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

tuned_parameters = [{'kernel': ['rbf'],
                     'C': [1]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=10,
                       scoring='%s_macro' % score)
    clf.fit(x_train, y_train_encoded)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    #y_train, y_pred = y_test, clf.predict(x_test)
    y_train2, y_pred = y_test_encoded, clf.predict(x_test)
    print(classification_report(y_train2, y_pred))
mid = df_3['readmitted']
df_3.drop(labels=['readmitted'], axis=1,inplace = True)
df_3.insert(0, 'readmitted', mid)
print(df_3.shape)


msk = np.random.rand(len(df_3)) < 0.8
df, df_test = df_3[msk].copy(deep = True), df_3[~msk].copy(deep = True)
df = df.reset_index()
df_test = df_test.reset_index()
print("df.shape",df.shape)
print("df_test.shape",df_test.shape)
y_train=df['readmitted']
y_test=df_test['readmitted']
print(set(y_train))
print(set(y_test))

x_train=df.iloc[:,1:50]
x_test=df_test.iloc[:,1:50]
print(set(x_train))
print(set(x_test))    print()


# In[ ]:

mid = df_3['readmitted']
df_3.drop(labels=['readmitted'], axis=1,inplace = True)
df_3.insert(0, 'readmitted', mid)
for i in range(0,df_3.shape[0]):
    if(df_3.loc[i,'readmitted']>0):
        df_3.loc[i,'solucio']=1
    else:
        df_3.loc[i,'solucio']=0
df_3.drop(labels=['readmitted'], axis=1,inplace = True)
print(df_3.shape)
print(df_3.columns)
df_4 = df_3.fillna(value=np.mean(df_3,axis=0),inplace=False,axis=0).values
print(df_4.shape)
print(df_4)
msk = np.random.rand(len(df_4)) < 0.8
df, df_test = df_4[msk].copy(), df_4[~msk].copy()
print("df.shape",df.shape)
print("df_test.shape",df_test.shape)
y_train=df[-1]
y_test=df_test[-1]
print(set(y_train))
print(set(y_test))

x_train=df.iloc[:,0:-1]
x_test=df_test.iloc[:,0:-1]
print(set(x_train))
print(set(x_test))

print ("data",data.shape)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[65]:

from imblearn.over_sampling import SMOTE
sm = SMOTE(n_jobs=-1, random_state=42,kind='regular')
x_train_res,y_train_res = sm.fit_sample(x_train,y_train)
x_test_res,y_test_res = sm.fit_sample(x_test,y_test)
print("Test", set(y_test))


# In[ ]:



