#!/usr/bin/env python
# coding: utf-8

# Using decision tree classification the factors responsible for 80%
# - occd_risk: High Risk 
# - ead_disposition: Past Due EAD
# - pmo_managed: No
# - net_ind: On-Net
# 
# Actionable Insights:
# - Develop a risk mitigation plan that includes specific actions to address high-risk areas and minimize potential negative outcomes
# - Implement strategies to reduce the number of past due EADs, such as offering payment plans or reminders for upcoming payments
# -  Implementing a project management office (PMO), which can help improve project management processes, enhance communication, and increase efficiency
# - Analyze the implications of being an on-net organization while exploring opportunities to leverage to increase effeciency
# - Develop a plan to optimize network performance

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[3]:


df = pd.read_csv('/Users/mac/Desktop/PROJECTS/WWFEMI/mccd_data.csv')
df


# In[4]:


df.drop('Column1', axis= 1)
df.head()


# In[5]:


df.columns = df.columns.str.lower()


# In[6]:


df.columns


# In[7]:


cat_ft= df[['bu_short', 'bu_segment', 'source_system','mrr_bucket', 'net_ind', 'order_activity', 'product_category',
       'product_ultimate_category']]


# In[8]:


import warnings
warnings.filterwarnings("ignore")
import textwrap


# In[9]:


per= df.apply(lambda x: x.value_counts(normalize=True)*100)


# In[10]:


import seaborn as sns
fig , ax = plt.subplots(4,2,figsize = (45,20))     # set up 2 x 2 frame count plot with figsize 10 x 10
for i , subplots in zip (cat_ft, ax.flatten()):  
  sns.countplot(cat_ft[i],hue = df['past_due_mccd'],ax = subplots, palette = 'BuPu')
plt.show()


# In[11]:


cat_ft= df[[ 'occd_risk','why_excluded_past_due_mccd','mccd_disposition', 'task_milestone', 
             'pmo_managed','network_build_flag','ltch_ever', 'is_affiliate']]


# In[12]:


fig , ax = plt.subplots(4,2,figsize = (45,20))     # set up 2 x 2 frame count plot with figsize 10 x 10
for i , subplots in zip (cat_ft, ax.flatten()):  
  sns.countplot(cat_ft[i],hue = df['past_due_mccd'],ax = subplots, palette = 'BuPu')
plt.show()


# In[13]:


cat_ft= df[[ 'build_category', 'cust_delay_ind', 'ecn_flag','sales_channel']]


# In[14]:


fig , ax = plt.subplots(2,2,figsize = (45,20))     # set up 2 x 2 frame count plot with figsize 10 x 10
for i , subplots in zip (cat_ft, ax.flatten()):  
  sns.countplot(cat_ft[i],hue = df['past_due_mccd'],ax = subplots, palette = 'BuPu')
plt.show()


# In[45]:


a= df.drop(['past_due_mccd','column1','mccd_disposition'], axis= 1)
y= df['past_due_mccd']


# In[46]:


pd.set_option('display.max_columns', None)
a.head(10)


# In[47]:


x= pd.get_dummies(a)


# In[48]:


type(y)


# In[54]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# In[55]:


#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[58]:


model = DecisionTreeClassifier()
model.fit(x, y)


# In[59]:


imp_scores = model.feature_importances_

total_imp = sum(imp_scores)
percentage_imp = [round(score/total_imp*100, 2) for score in imp_scores]


# In[60]:


feature_imp = dict(zip(x.columns, percentage_imp))

sorted_ft = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)


# In[61]:



cumulative_percentage = 0
for feature in sorted_ft:
    cumulative_percentage += feature[1]
    if cumulative_percentage >= 80:
        break
        
print([feature[0] for feature in sorted_ft[:sorted_ft.index((feature[0], feature[1]))+1]])


# **** Another approach***

# In[62]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[63]:


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# In[64]:


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


# In[65]:


importance = model.feature_importances_


# In[66]:


sorted_idx = importance.argsort()[::-1]
cumulative_importance = np.cumsum(importance[sorted_idx])


# In[67]:


index_80 = np.where(cumulative_importance > 0.8)[0][0]
factors_80 = x.columns[sorted_idx[:index_80+1]]


# In[68]:


print(factors_80)


# In[71]:


df['net_ind'].value_counts()


# In[ ]:




