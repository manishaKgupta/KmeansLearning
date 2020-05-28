#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
dataset=pd.read_csv('in.csv')
dataset=dataset.iloc[:,[1,2]]
lat=dataset['lat']
long=dataset['lng']


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
sc = StandardScaler()
data_scaled=sc.fit_transform(dataset)
model=KMeans(n_clusters=5)
model_class = model.fit_predict(data_scaled)
model_pred = model.fit(data_scaled)
centroid = model.cluster_centers_


# In[3]:


data_scaled=pd.DataFrame(data_scaled, columns=['lat', 'lng'])
data_scaled['cluster class']=model_class
centroid = pd.DataFrame(centroid,columns=['lat','lng'])


# In[4]:


import plotly.graph_objs as go
import plotly.offline as pyo
data = [go.Scatter(x=data_scaled['lat'],
                   y=data_scaled['lng'].where(data_scaled['cluster class']==c),
                   mode='markers')for c in range(5)]
data.append(go.Scatter(x=centroid['lat'],
                       y=centroid['lng'],
                       mode='markers',
                       marker=dict(size=10,color='#000000')))
pyo.plot(data,filename='output.html')

