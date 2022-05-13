#!/usr/bin/env python
# coding: utf-8

# Aim: Devising a trading startegy for forex pairs based on continental based market opening and closing timings. Taking into account Volume data and conducting analysis using (1) Prediction Models and (2) Classification Models. 
# 
# Classification is based of a binary variables that is 1 if European Close > Asian Close.
# 
# Models making use of :
# * Prediction
# (a) Multiple Linear Regression Model
# (b) Deep Nueral Network
# 
# 
# * Classification
# (a) Logistic Regression
# (b) SVM
# (c) Deep Nueral Network (Classification version)

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df_1 = pd.read_csv("C:/Users/mihirgupta/OneDrive - Deloitte (O365D)/Forex Data/GBPJPY60.csv")
df_2 = pd.read_csv("C:/Users/mihirgupta/OneDrive - Deloitte (O365D)/Forex Data/GBPJPY_AlgoData.csv")


# In[4]:


#Date Time Adjustments 
df_1['Hour']=pd.to_datetime(df_1["Hour"])
df_1['Date']=pd.to_datetime(df_1["Date"])
df_1['Hour'] = df_1.Hour.apply(lambda x: x.hour)

#Extract Day
df_1["Week Day"] = df_1["Date"].dt.weekday
df_1["Month"] = df_1["Date"].dt.month


# In[8]:


# Check data head
df_1.head()
df_2.head()


# In[7]:


# For the prediction model there are three statistics of interest 

#(a) Europen Close Value - Given in the data
#(b) Differ. between EC and AC
df_2['Difference'] = df_2['EC']-df_2['AC']

#(c) Return between EC and AC
df_2['Return'] = (df_2['EC']-df_2['AC'])/df_2['AC']


# In[9]:


#Initial Analysis suggests that indeed there may be some predictive power in AO, AC, EO in determining the value of EC

plt.figure(figsize=(12,6))
plt.plot( 'Day.No', 'AC', data=df_2, color='blue', linewidth=2)
plt.plot( 'Day.No', 'EC', data=df_2, color='red', linewidth=2)

# show legend
plt.legend()

# show graph
plt.show()

"\n"

plt.figure(figsize=(12,6))
plt.plot( 'Day.No', 'AO', data=df_2, color='blue', linewidth=2)
plt.plot( 'Day.No', 'EC', data=df_2, color='red', linewidth=2)

# show legend
plt.legend()

# show graph
plt.show()

"\n"

plt.figure(figsize=(12,6))
plt.plot( 'Day.No', 'EO', data=df_2, color='blue', linewidth=2)
plt.plot( 'Day.No', 'EC', data=df_2, color='red', linewidth=2)

# show legend
plt.legend()

# show graph
plt.show()


# In[10]:


#Volume across the timings are not that responsive to one another
plt.figure(figsize=(12,6))
plt.plot( 'Day.No', 'Vol_AC', data=df_2, color='orange', linewidth=2)
plt.plot( 'Day.No', 'Vol_EC', data=df_2, color='green', linewidth=2)

# show legend
plt.legend()

# show graph
plt.show()

"\n"

#Volume across the timings are not that responsive to one another
plt.figure(figsize=(12,6))
plt.plot( 'Day.No', 'Vol_AO', data=df_2, color='orange', linewidth=2)
plt.plot( 'Day.No', 'Vol_EC', data=df_2, color='green', linewidth=2)

# show legend
plt.legend()

# show graph
plt.show()

"\n"

#Volume across the timings are not that responsive to one another
plt.figure(figsize=(12,6))
plt.plot( 'Day.No', 'Vol_EO', data=df_2, color='orange', linewidth=2)
plt.plot( 'Day.No', 'Vol_EC', data=df_2, color='green', linewidth=2)

# show legend
plt.legend()

# show graph
plt.show()


# In[11]:


#Density Plot for the Volumes

plt.figure(figsize=(12,6))
sns.kdeplot(df_2['Vol_AO'], color="red")
"\n"
sns.kdeplot(df_2['Vol_AC'], color="blue")
"\n"
sns.kdeplot(df_2['Vol_EO'], color="orange")
"\n"
sns.kdeplot(df_2['Vol_EC'], color="green")


# * There is a leftward shift in the peak of the volume densities - starting from AO -> EC.
# * There is a momentum push in volume with time

# In[12]:


#Density Plot for the Volumes

plt.figure(figsize=(12,6))
sns.kdeplot(df_2['AO'], color="red")
"\n"
sns.kdeplot(df_2['AC'], color="blue")
"\n"
sns.kdeplot(df_2['EO'], color="orange")
"\n"
sns.kdeplot(df_2['EC'], color="green")


# In[17]:


sns.displot(df_2['Difference'], kde=True)
"\n"
#sns.displot(df_2['Return'], kde=True)

#Essentially the same plot - just factored out by the denominator in case of Return
#Therefore we can reduce our prediction variable to just EC and Difference/Return


# * Prediction Models - Linear Regression

# In[42]:


#Linear Regression - Prediction
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[38]:


#Determining variables
X = df_2[['AO', 'AC', 'EO','Vol_AO', 'Vol_AC','Vol_EO']]
y1 = df_2['EC']
y2 = df_2['Return']


# In[41]:


# Fitting Linear Regression with EC as dependent variable

X_full = sm.add_constant(X)
lin_reg = sm.OLS(y1, X_full)
lr = lin_reg.fit()
print(lr.summary())


# Conclusions: Essentially all varibales except AC are insignificant on an individual basis but the F-Stat is quite high.

# In[24]:


# Fitting Linear Regression with EC as dependent variable
#X_full = sm.add_constant(X)
lin_reg = sm.OLS(y2, X)
lr = lin_reg.fit()
print(lr.summary())


# Conclusion: Horrible Model - Not worth fitting 

# In[27]:


sns.jointplot(x='AC', y='EC', data=df_2, kind='reg')


# In[28]:


#Creating Test Train Split and plotting Predictions vs Real Values
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.4, random_state=101)
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)


# In[37]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
#There are differences in the coeffecient values due to use of different codes. However, as in the OLS case the wider 
#result of Volumes having negligible impact and AC having the highest coeff. match. 


# In[50]:


print(lm.intercept_)


# In[49]:


#Residual Plot
sns.displot((y_test-predictions),bins=20,kde=True)


# In[48]:


#Printing Regression Evaluation Metrics
print('EC Mean:', df_2['EC'].mean())
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# * Prediction Model - Deep Neural Network
