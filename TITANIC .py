#!/usr/bin/env python
# coding: utf-8

# # Titanic Dataset: exploratory data analysis and feature engineer

# In this notebook, we're going to analyes the titanic dataset,but we are going to do explorarity analysis.

# In[1]:


# importing necessary libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


Dataset=pd.read_csv("C:/Users/computer/Desktop/Data set/titanic.csv")


# In[3]:


Dataset.head()


# In[4]:


#information about the dataset.
Dataset.info()


# We see that the dataset is missing lots of information in Cabin column

# In[5]:


#The describe() method shows a summary of the numerical attributes
Dataset.describe()


# # Exploratory Data  Analysis

# In[6]:


#checking wheather my dataset is balanced or imbalanced
Dataset["Survived"].value_counts()


# Graphic Representation 

# In[7]:


count_classes = pd.value_counts(Dataset['Survived'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("No of Survived")

plt.xlabel("Survived")

plt.ylabel("counts")


# this is balanced dataset

# Now checking on how the passengers were divided among different classes.

# In[8]:


sns.factorplot('Pclass',data=Dataset,hue='Sex',kind='count')


# In[9]:


sns.pairplot(Dataset)


# checking the person is a man,women,child

# In[10]:


Dataset["Pclass"].value_counts()


# In[11]:


def man_women_Child(passenger):
    age=passenger['Age']
    sex=passenger['Sex']
    return 'Child' if age < 16 else sex
#create a new column  person
Dataset["Person"]=Dataset.apply(man_women_Child,axis=1)


# In[12]:


Dataset.head()


# In[13]:


#NOW SEEING COUNTS IN NEW COLUMN PERSON


# In[14]:


Dataset["Person"].value_counts()


# In[15]:


sns.factorplot('Pclass',data=Dataset,hue='Person',kind='count')


# In[16]:


Dataset["Age"].hist()


# # Droping Null Values

# In[17]:


Titanic_df=Dataset.dropna(axis=0)


# In[18]:


Titanic_df.info()


# In[19]:


#Grabbing the deck from the cabin numbers
def get_lvl(passenger):
    cabin = passenger['Cabin']
    return cabin[0]


# In[20]:


Titanic_df["lvl"]=Titanic_df.apply(get_lvl,axis=1)


# In[21]:


Titanic_df.head()


# In[23]:


sns.factorplot('lvl',data=Titanic_df,palette='winter_d',kind='count')


# Checking how many passenger belong to different decks

# Passengers with Family or Alone

# In[24]:


sns.factorplot('lvl',data=Titanic_df,hue='Pclass',kind='count')


# # Where did the passengers come from

# The Embarked attribute contains data for the passengers' port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).

# In[29]:


sns.factorplot('Embarked',data=Titanic_df,hue='Pclass',kind='count')


# In[33]:


Titanic_df.head()


# Passengers with Family and Alone

# In[39]:


Titanic_df['Alone'] = Titanic_df.SibSp + Titanic_df.Parch


# In[40]:


Titanic_df["Alone"].value_counts()


# In[45]:


sns.factorplot('Alone',data=Titanic_df,kind='count')


# In[49]:


sns.factorplot('Alone','Survived',data=Titanic_df)


# # FEATURE ENGINEERING

# In[50]:


df=pd.read_csv("C:/Users/computer/Desktop/Data set/titanic.csv")


# In[51]:


df.head()


# In[52]:


df.shape


# In[54]:


df.isnull().sum()


# In[55]:


df["Age"].hist()


# In[56]:


def impute_nan(df,variable,median):
    df[variable+"_median"]=df[variable].fillna(median) 


# In[57]:


median=df['Age'].median()


# In[58]:


median


# In[60]:


impute_nan(df,'Age',median)
df.head()


# In[66]:


df["cabin_nan"]=np.where(df["Cabin"].isnull(),1,0)


# In[67]:


df.head()


# In[68]:


df=df.drop(["Name","Ticket","Cabin",],axis=1)


# In[69]:


df


# In[71]:


df.Embarked.value_counts().unique()


# In[88]:


df=pd.get_dummies(df,drop_first=True).head()


# In[89]:


df


# In[77]:


print(df['Age'].std())
print(df['Age_median'].std())


# In[78]:


fig = plt.figure()
ax = fig.add_subplot(111)
df['Age'].plot(kind='kde', ax=ax)
df.Age_median.plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')


# In[83]:


df.drop(["Age"],axis=1).head()


# In[84]:


df.isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:




