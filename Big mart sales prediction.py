#!/usr/bin/env python
# coding: utf-8

# In[44]:


#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


# In[2]:


data=pd.read_csv("train.csv")


# In[3]:


data


# In[4]:


#checking null values
data.isnull().sum()


# In[5]:


data["Item_Fat_Content"].value_counts()


# In[6]:


data["Item_Fat_Content"].replace({"LF":"Low Fat","reg":"Regular","low fat":"Low Fat"},inplace=True)


# In[7]:


#Filling Nan values with mean
data["Item_Weight"].fillna(data["Item_Weight"].mean(),inplace=True)


# In[8]:


data["Outlet_Size"].fillna(0,inplace=True)


# In[9]:


#Filling Nan values with mode
data["Outlet_Size"].replace({0:"Medium"},inplace=True)


# In[10]:


data["Outlet_Size"].value_counts()


# In[11]:


data.isnull().sum()


# In[12]:


n=[]
for i in range(len(data)):
    n.append(data["Item_Identifier"][i][0:2])
new=pd.DataFrame({"Item_Type_Combined":n})
new
df=pd.concat([new,data],axis=1)


# In[13]:


df["Item_Type_Combined"].replace({"FD":"Food","DR":"Drinks","NC":"Non-Consumable"},inplace=True)


# In[14]:


#If Item_Type_Combined is non-consumable then we are replacing item_fat_content as non-edible
df.loc[df["Item_Type_Combined"]=="Non-Consumable","Item_Fat_Content"]="Non-Edible"
df["Item_Fat_Content"].value_counts()


# In[15]:


#since this data is taken at the year of 2013
df["Outlet_Year"]=2013-df["Outlet_Establishment_Year"]


# In[16]:


df


# In[17]:


#univariant analysis
sns.distplot(df["Item_Weight"])


# In[18]:


sns.distplot(df["Item_Visibility"])


# In[19]:


sns.distplot(df["Item_MRP"])


# In[20]:


sns.distplot(df["Item_Outlet_Sales"])


# In[21]:


#since the outlet sales is right skewed we are doing Log transformation
df["Item_Outlet_Sales"]=np.log(1+df["Item_Outlet_Sales"])


# In[22]:


sns.distplot(df["Item_Outlet_Sales"])


# In[23]:


sns.countplot(df["Item_Fat_Content"])


# In[24]:


sns.countplot(df["Outlet_Size"])


# In[25]:


sns.countplot(df["Outlet_Location_Type"])


# In[26]:


plt.figure(figsize=(10,6))
sns.countplot(df["Outlet_Type"])


# In[27]:


plt.figure(figsize=(24,6))
sns.countplot(df["Item_Type"])


# In[28]:


sns.countplot(df["Outlet_Establishment_Year"])


# In[29]:


#Bivariant analysis
plt.figure(figsize=(10,8))
plt.xlabel("Item_Weight")
plt.ylabel("Item_Outlet_Sales")
plt.plot(df["Item_Weight"],df["Item_Outlet_Sales"],".",alpha=0.3)


# In[30]:


plt.figure(figsize=(10,8))
plt.xlabel("Item_Visibility")
plt.ylabel("Item_Outlet_Sales")
plt.plot(df["Item_Visibility"],df["Item_Outlet_Sales"],".",alpha=0.3)


# In[31]:


Item_Fat_Content_pivot =df.pivot_table(index='Item_Fat_Content', values="Item_Outlet_Sales", aggfunc=np.median)
Item_Fat_Content_pivot.plot(kind='bar', color='green',figsize=(12,7))
plt.xlabel("Item_Fat_Content")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Item_Fat_Content on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[32]:


Outlet_Establishment_Year_pivot =df.pivot_table(index='Outlet_Establishment_Year', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Establishment_Year_pivot.plot(kind='bar', color='green',figsize=(12,7))
plt.xlabel("Outlet_Establishment_Year")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Establishment_Year on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[33]:


Outlet_Identifier_pivot =df.pivot_table(index='Outlet_Identifier', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Identifier_pivot.plot(kind='bar', color='green',figsize=(12,7))
plt.xlabel("Outlet_Identifier")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Identifier on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[34]:


Outlet_Size_pivot =df.pivot_table(index='Outlet_Size', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Size_pivot.plot(kind='bar', color='green',figsize=(12,7))
plt.xlabel("Outlet_Sizer")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Size on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[35]:


Outlet_Location_Type_pivot =df.pivot_table(index='Outlet_Location_Type', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Location_Type_pivot.plot(kind='bar', color='green',figsize=(12,7))
plt.xlabel("Outlet_Location_Type")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Location_Type on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[36]:


Outlet_Type_pivot =df.pivot_table(index='Outlet_Type', values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Type_pivot.plot(kind='bar', color='green',figsize=(12,7))
plt.xlabel("Outlet_Type")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Type on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()


# In[37]:


#Let's check the correlation between the numerical values
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm")


# In[38]:


#Doing label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Outlet"]=le.fit_transform(df["Outlet_Identifier"])
cat_colms=["Item_Fat_Content","Item_Type","Outlet_Size","Outlet_Location_Type","Outlet_Type","Item_Type_Combined"]
for c in cat_colms:
    df[c]=le.fit_transform(df[c])


# In[39]:


#Deleting the columns which are already processed
new=df.drop(["Outlet_Identifier","Outlet_Establishment_Year","Item_Identifier"],axis=1)
new


# In[40]:


#Doing one hot encoding
new=pd.get_dummies(new,columns=["Item_Fat_Content","Outlet_Size","Outlet_Location_Type","Outlet_Type","Item_Type_Combined"])


# In[41]:


new.shape


# In[42]:


#Defining variables
x=new.drop("Item_Outlet_Sales",axis=1)
y=new["Item_Outlet_Sales"]


# In[43]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[59]:


#Traing the model
model=LinearRegression()
model.fit(x_train,y_train)
print("cross validation score :",np.mean(cross_val_score(model,x_train,y_train)))
print("model score :",model.score(x_test,y_test))
y_pred=model.predict(x_test)
print("MSE :",mean_squared_error(y_test,y_pred))


# In[56]:


tree=DecisionTreeRegressor()
tree.fit(x_train,y_train)
print("cross validation score :",np.mean(cross_val_score(tree,x_train,y_train)))
print("model score :",tree.score(x_test,y_test))


# In[55]:


xgb=XGBRegressor()
xgb.fit(x_train,y_train)
print("cross validation score :",np.mean(cross_val_score(xgb,x_train,y_train)))
score=xgb.score(x_test,y_test)
print("model score : ",score)

