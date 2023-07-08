#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import chardet
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans


# In[7]:


df = pd.read_excel('Online Retail.xlsx')


# In[10]:


df.to_csv('/Users/sanan33/Desktop/Customers.csv',index=False)


# In[24]:


df = pd.read_csv('/Users/sanan33/Desktop/Customers.csv')


# In[25]:


df


# In[26]:


df.info()


# In[27]:


df.isna().sum()


# In[28]:


df = df.dropna()


# In[29]:


df.isna().sum()


# In[30]:


df['TotalPrice'] = df['Quantity']*df['UnitPrice']


# In[31]:


df[df['TotalPrice']<=0]


# In[32]:


df=df.drop(df[df['TotalPrice']<=0].index)


# In[33]:


df


# In[38]:


px.box(df['TotalPrice'])


# In[39]:


q1 = df['TotalPrice'].quantile(0.25)
q3 = df['TotalPrice'].quantile(0.75)
iqr = q3-q1
lower = q1-1.5*iqr
upper = q3+1.5*iqr


# In[40]:


df = df[~((df['TotalPrice']>upper)| (df['TotalPrice']<lower))]


# In[41]:


px.box(df['TotalPrice'])


# In[44]:


df = df.reset_index(drop=True)
df


# In[45]:


df.drop(['level_0','index'],axis=1,inplace=True)


# In[49]:


df.nunique()   #method1


# In[50]:


len(df['CustomerID'].unique())    #method2


# In[51]:


df['InvoiceNo'].value_counts()


# In[54]:


df['CustomerID']=df['CustomerID'].astype('int')


# In[55]:


df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


# In[56]:


df.info()


# In[58]:


today = df['InvoiceDate'].max()
today


# In[62]:


today = dt.datetime(2011,12,9,12,50,0)
today


# ## Recency

# In[63]:


r = (today-df.groupby('CustomerID').agg({'InvoiceDate':'max'})).apply(lambda x:x.dt.days)
r


# ## Frequency

# In[69]:


f1 = df.groupby(['CustomerID','InvoiceNo']).agg({'InvoiceNo':'count'})
f = f1.groupby('CustomerID').agg({'InvoiceNo':'count'})
print(f)
print(f1)


# ## Monetary

# In[71]:


m = df.groupby('CustomerID').agg({'TotalPrice':'sum'})
m


# In[78]:


RFM = r.merge(f,on='CustomerID').merge(m,on='CustomerID')
RFM = RFM.reset_index()
RFM = RFM.rename({'InvoiceDate':'Recency','InvoiceNo':'Frequency','TotalPrice':'Monetary'},axis=1)
RFM


# ## Clustering

# In[79]:


df = RFM.iloc[:,1:]
df


# In[83]:


sc = MinMaxScaler()
dfnorm = sc.fit_transform(df)
dfnorm = pd.DataFrame(dfnorm,columns=df.columns)
dfnorm


# In[85]:


model = KMeans()
#graph = KElbowVisualizer(model,k=(2,10))
#graph.fit(dfnorm)
#graph.poof()


# In[86]:


model = KMeans(n_clusters=4,init = 'k-means++')
kfit = model.fit(dfnorm)
labels = kfit.labels_


# In[88]:


sns.scatterplot(x = 'Recency',y='Frequency',data=dfnorm,hue=labels,palette='deep')


# In[89]:


RFM['Labels']=labels
print(RFM)


# In[93]:


sns.scatterplot(x = 'Labels',y='CustomerID',data=RFM,hue=labels,palette='deep')
plt.xlim(-1,4)


# In[94]:


RFM.groupby('Labels')['CustomerID'].count()


# In[96]:


RFM.groupby('Labels').mean().iloc[:,1:]


# In[98]:


#Number 2 group have low recency,frequency and monetary.This group lost customer group
#Best group is Number 3


# In[ ]:




