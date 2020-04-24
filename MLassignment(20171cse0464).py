#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import seaborn as sns # for data visualization
import matplotlib.pyplot as plt # for data visualization
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("c:/Users/admin/Downloads/Breast_cancer_data.csv", delimiter=",")


# In[4]:


df.head() #gives first 5 entries of a dataframe by default


# In[5]:


df.columns


# In[6]:


df.isnull().sum()


# In[7]:


count = df.diagnosis.value_counts()
count


# In[8]:


count.plot(kind='bar')
plt.title("Distribution of malignant(1) and benign(0) tumor")
plt.xlabel("Diagnosis")
plt.ylabel("count");


# In[9]:


y_target = df['diagnosis']


# In[10]:


df.columns.values


# In[11]:


df['target'] = df['diagnosis'].map({0:'B',1:'M'}) # converting the data into categorical


# In[31]:


g = sns.pairplot(df.drop('diagnosis', axis = 1), hue="target", palette='prism');


# In[24]:


sns.scatterplot(x='mean_perimeter', y = 'mean_texture', data = df, hue = 'target', palette='prism');


# In[29]:


features = ['mean_perimeter', 'mean_texture']


# In[30]:


X_feature = df[features]


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


X_train, X_test, y_train, y_test= train_test_split(X_feature, y_target, test_size=0.3, random_state = 42)


# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[35]:


model = LogisticRegression()


# In[36]:


model.fit(X_train, y_train)


# In[43]:


y_pred = model.predict(X_test)


# In[44]:


acc = accuracy_score(y_test, y_pred)
print("Accuracy score using Logistic Regression:", acc*100)


# In[48]:


from sklearn.metrics import confusion_matrix


# In[49]:


conf_mat = confusion_matrix(y_test, y_pred)


# In[50]:


conf_mat


# In[51]:


from sklearn.neighbors import KNeighborsClassifier


# In[52]:


clf = KNeighborsClassifier()


# In[53]:


clf.fit(X_train, y_train)


# In[54]:


y_pred = clf.predict(X_test)


# In[55]:


acc = accuracy_score(y_test, y_pred)
print("Accuracy score using KNN:", acc*100)


# In[56]:


confusion_matrix(y_test, y_pred)


# In[ ]:




