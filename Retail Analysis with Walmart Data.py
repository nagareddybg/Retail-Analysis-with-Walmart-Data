#!/usr/bin/env python
# coding: utf-8

# # Retail Analysis with Walmart Data

# # Analysis Tasks
# 
# 1. Basic Statistics tasks
# 
# 1. 1 Which store has maximum sales
# 
# 1. 2 Which store has maximum standard deviation i.e., the sales vary a lot. Also, find out the coefficient of mean to standard deviation
# 
# 1. 3 Which store/s has good quarterly growth rate in Q3’2012
# 
# 1. 4 Some holidays have a negative impact on sales. Find out holidays which have higher sales than the mean sales in non-holiday season for all stores together
# 
# 1. 5 Provide a monthly and semester view of sales in units and give insights
# 
# 2. Statistical Model
# 
# For Store 1 – Build  prediction models to forecast demand
# 
# 2. 1 Linear Regression – Utilize variables like date and restructure dates as 1 for 5 Feb 2010 (starting from the earliest date in order). Hypothesize if CPI, unemployment, and fuel price have any impact on sales.
# 
# 2. 2 Change dates into days by creating new variable.

# In[117]:


# Import libraries

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
from datetime import datetime


# In[6]:


# Load dataset
data = pd.read_csv('D:\\walmart_store_sales\\Walmart_Store_sales.csv')
data.head()


# In[3]:


# check shape of data
data.shape


# In[7]:


data.info()


# In[8]:


da


# In[9]:


data.Store.unique()


# 1. 1 Which store has maximum sales

# In[10]:


sales_list=[]
sales_list=data.groupby(['Store'])['Weekly_Sales'].sum()
max_sales=max(data.groupby(['Store'])['Weekly_Sales'].sum())
sales_list


# In[11]:


for i in range(1,46):
    if max_sales==sales_list[i]:
        print("Store which has maximum sales of {} is {}".format(max_sales,i))

# Conclusion
Store 20 has maximum sales of 301397792.46
# 1. 2 Which store has maximum standard deviation 

# In[14]:


std_dev=[]
std_dev=data.groupby(['Store'])['Weekly_Sales'].std()
max_std=max(data.groupby(['Store'])['Weekly_Sales'].std())
print(std_dev)


# In[ ]:





# In[15]:


for i in range(1,46):
    if max_std==std_dev[i]:
        print('Store which has maximum standard deviation of {} is {}'.format(max_std,i))


# Store 14 has maximum standard deviation of 317569.9494755081

# 1. 3 Which store/s has good quarterly growth rate in Q3’2012

# In[32]:


data_safe=data


# In[33]:


data["Date"]=pd.to_datetime(data["Date"])


# In[35]:


#Third Quartile Period

date_from=pd.Timestamp(date(2012,7,1))
date_to = pd.Timestamp(date(2012,9,1))


# In[36]:


data_safe = data_safe[
    (data_safe['Date'] > date_from ) &
    (data_safe['Date'] < date_to)]


# In[37]:


data_safe


# In[40]:


Q3_growth=[]
Q3_growth=data_safe.groupby(['Store'])['Weekly_Sales'].sum()
max_Q3_growth=max(data_safe.groupby(['Store'])['Weekly_Sales'].sum())
print(Q3_growth)
print(max_Q3_growth)


# In[56]:


max_Q3_growth

# Conclusion

Store 14 has maximum Q3 growth of 17184755.18
# 1. 4 Some holidays have a negative impact on sales.

# In[58]:


Christmas_sales=data.loc[(data["Date"]=="2010-12-31") | (data["Date"]=="2011-12-31") | (data["Date"]=="2012-12-28") | (data["Date"]=="2013-12-")]
Christmas_sales.head()


# In[57]:


Christmas_sales["Weekly_Sales"].sum()


# In[61]:


Labour_Day=data.loc[(data["Date"]=="2010-09-10") | (data["Date"]=="2011-09-09") | (data["Date"]=="2012-09-07") | (data["Date"]=="2013-09-06")]
Labour_Day.head()


# In[63]:


Labour_Day["Weekly_Sales"].sum()


# In[65]:


Thanksgivings=data.loc[(data["Date"]=="2010-11-26") | (data["Date"]=="2011-11-25") | (data["Date"]=="2012-11-23") | (data["Date"]=="2013-11-29")]
Thanksgivings.head()


# In[66]:


Thanksgivings["Weekly_Sales"].sum()


# Conclusion
# 
# 1. Holidays which have higher sales is Thanksgivings
# 2. Total weekly sales of thanksgiving holidays is 132414608.5
# 3. Total sales in Labour day is 46763227.5
# 4. Total sales in christmas holidays is 40432519.0

# 1. 5 Provide a monthly and semester view of sales in units and give insights

# In[68]:


data["Year"]= pd.DatetimeIndex(data['Date']).year
data["Month"]= pd.DatetimeIndex(data['Date']).month


# In[71]:


year_2010=data.loc[data["Year"]==2010]
year_2011=data.loc[data["Year"]==2011]
year_2012=data.loc[data["Year"]==2012]


# In[76]:


# Monthly view of sales in 2010

plt.figure(figsize=(10,6))
plt.bar(year_2010["Month"],year_2010["Weekly_Sales"])
plt.xlabel("months")
plt.ylabel("Weekly Sales")
plt.title("Monthly view of sales in 2010")


# In[77]:


# Monthly view of sales in 2011
plt.figure(figsize=(10,6))
plt.bar(year_2011["Month"],year_2011["Weekly_Sales"])
plt.xlabel("months")
plt.ylabel("Weekly Sales")
plt.title("Monthly view of sales in 2011")


# In[78]:


# Monthly view of sales in 2012

plt.figure(figsize=(10,6))
plt.bar(year_2012["Month"],year_2012["Weekly_Sales"])
plt.xlabel("months")
plt.ylabel("Weekly Sales")
plt.title("Monthly view of sales in 2012")


# In[82]:


# All months view of sales

plt.figure(figsize=(12,8))
plt.bar(data["Month"],data["Weekly_Sales"])
plt.xlabel("months")
plt.ylabel("Weekly Sales")
plt.title("Monthly view of sales")


# In[83]:


# Semesterwise sale 
semester_sales=[]
semester_sales.append(year_2010.loc[year_2010["Month"]<7,["Weekly_Sales"]].sum())
semester_sales.append(year_2010.loc[year_2010["Month"]>6,["Weekly_Sales"]].sum())
semester_sales.append(year_2011.loc[year_2011["Month"]<7,["Weekly_Sales"]].sum())
semester_sales.append(year_2011.loc[year_2011["Month"]>6,["Weekly_Sales"]].sum())
semester_sales.append(year_2012.loc[year_2012["Month"]<7,["Weekly_Sales"]].sum())
semester_sales.append(year_2012.loc[year_2012["Month"]>6,["Weekly_Sales"]].sum())


# In[84]:


semester_names=["sem1_2010","sem2_2010","sem1_2011","sem2_2011","sem1_2012","sem2_2012"]


# In[85]:


plt.figure(figsize=(10,6))
plt.plot(semester_names,semester_sales)
plt.xlabel("Semesters")
plt.ylabel("Semester Sales")
plt.title("Semester view of sales")


# 2. Statistical Model
# For Store 1 – Build  prediction models to forecast demand
# 2. 1 Linear Regression – Utilize variables like date and restructure dates as 1 for 5 Feb 2010 (starting from the earliest date in order). Hypothesize if CPI, unemployment, and fuel price have any impact on sales.

# In[86]:


x=data.drop(["Weekly_Sales","Date"],axis=1)
y=data["Weekly_Sales"]


# In[87]:


linreg=LinearRegression(n_jobs=-1)


# In[97]:


xtrain,xtest,ytrain,ytest=model_selection.train_test_split(x,y,test_size=0.4,random_state=42)


# In[98]:


linreg.fit(xtrain,ytrain)


# In[99]:


linreg.intercept_


# In[100]:


linreg.coef_


# In[101]:


x.columns


# In[102]:


features=['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month'],


# In[103]:


relation=pd.Series(linreg.coef_,x.columns).sort_values()
relation.plot(kind="bar")


# In[104]:


linreg.score(xtest,ytest)


# In[106]:


sqrt(mean_squared_error(ytrain,linreg.predict(xtrain)))


# In[107]:


mean_squared_error(ytest,linreg.predict(xtest))


# In[110]:


#Relation Between CPI and weekly sales

plt.figure(figsize=(8,6))
plt.scatter(data["CPI"],data["Weekly_Sales"])
plt.title("Relation Between CPI and weekly sales")
plt.xlabel("CPI")
plt.ylabel("Weekly Sales")


# In[112]:


#Relation Between Unemployment and weekly sales

plt.figure(figsize=(8,6))
plt.scatter(data["Unemployment"],data["Weekly_Sales"])
plt.title("Relation Between Unemployment and weekly sales")
plt.xlabel("Unemployment")
plt.ylabel("Weekly Sales")


# In[114]:


#Relation Between Fuel Price and weekly sales
plt.figure(figsize=(8,6))
plt.scatter(data["Fuel_Price"],data["Weekly_Sales"])
plt.title("Relation Between Fuel Price and weekly sales")
plt.xlabel("Fuel_Price")
plt.ylabel("Weekly Sales")


# 2. 2 Change dates into days by creating new variable.

# In[115]:


data['days'] = data['Date'].dt.day_name()


# In[116]:


data

