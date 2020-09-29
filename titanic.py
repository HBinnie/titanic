#Kaggle - Titanic Dataset Competition

#Introduction: This project will aim to predict the likelihood of survival of an individual travelling on the Titanic. To do so, we will utilise a range of features including their Class Ticket Status, their Name and family members aboard and age among others.


#To begin, we first want to prepare all module imports which may be utilised in the project.

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

#In this case, the datafiles we are using are already broken up into train and test files. As such, we can simply load them directly using Pandas.

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#To get an idea about what the data contains, we can view the first 5 rows of the data. For now, we will focus on the df_train file as this contains the data will be using the develop our predictive model with.

df_train.head()

#Here we can view the columns within the dataset, alongside some examples of how the data appears. In the "Cabin" column, on lines 0, 2 and 4 we can see "NaN", which means 'not a number'. This suggests that the full dataset has some missing values.

#Another simple way to get some insight into the dataset is to use the 'describe' method.

df_train.describe()

#We can see that this method results in fewer columns, as it is only designed to handle numerical values, and categorical values are omitted.

# Lets explore the data a little further to understand each variable better.

tot_surv_rate = df_train['Survived'].mean()

#This is the total surrival rate.

print("The total survival rate of the Titanic sample is %.2f" %tot_surv_rate)

train_sex_dummies = pd.get_dummies(df_train['Sex'])
#train_cab_dummies = pd.get_dummies(df_train['Cabin'])
train_emb_dummies = pd.get_dummies(df_train['Embarked'])

df_train = pd.concat([df_train, train_sex_dummies, train_emb_dummies],axis = 1)

df_train = df_train.drop(['Sex','Cabin','Embarked','Name','PassengerId','Ticket'], axis = 1)
df_train = df_train.dropna(axis = 0)

X_train = df_train.drop(columns='Survived')
y_train = df_train['Survived']

clf = LogisticRegression(random_state = 0)

model = clf.fit(X,y)

#Prepare the Test data in the same manner.
test_sex_dummies = pd.get_dummies(df_test['Sex'])
#test_cab_dummies = pd.get_dummies(df_test['Cabin'])
test_emb_dummies = pd.get_dummies(df_test['Embarked'])

df_test = pd.concat([df_test, test_sex_dummies, test_emb_dummies],axis = 1)

df_test = df_test.drop(['Sex','Cabin','Embarked','Name','PassengerId','Ticket'], axis = 1)
df_test = df_test.dropna(axis = 0)

X_test = df_train.drop(columns='Survived')

y_pred = clf.predict(X_test)

score =clf.score(X_train, y_train)

print(y_train)
