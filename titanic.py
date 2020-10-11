<<<<<<< HEAD
# Kaggle - Titanic Dataset Competition

# Imports

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

from xgboost import XGBClassifier

# Set Path

path = Path('C:/Users/hbinn/Documents/Projects/kaggle/titanic/')

# Load Data

df_train = pd.read_csv(path/'train.csv')
df_test = pd.read_csv(path/'test.csv')

# Data Analysis & Exploration **TBA**

print(df_train.isnull().any())
print(df_test.isnull().any())



print("The total survival rate of the Titanic sample is %.4f" %df_train['Survived'].mean())

# Feature Engineering and Transfomation

    # Modifying Variables

df_train['Cabin'] = df_train['Cabin'].str.get(0)
df_train['Cabin'] = df_train['Cabin'] + "_cabin"



    # Creating Dummy variables for Categorical Data
train_sex_dummies = pd.get_dummies(df_train['Sex'])
train_emb_dummies = pd.get_dummies(df_train['Embarked'])
train_cabin_dummies = pd.get_dummies(df_train['Cabin'])

df_train = pd.concat([df_train, train_sex_dummies, train_emb_dummies,
                      train_cabin_dummies], axis =1)

    # Data Imputation for Missing Variables **TBA**


X = df_train

    # Data Scaling **TBA**

    # Removing unused variables and finalising data before fitting to model.

X = X.drop(columns =(['PassengerId','Name','Sex','Ticket','Cabin','Embarked','T_cabin']))

X = X.dropna()
y = X['Survived']
X = X.drop(columns ='Survived')

# Create Train & Validation Data Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)

# Prepare and fit the models we will assess.
    
    # Linear Regression
lin_reg = LinearRegression()
lin_model = lin_reg.fit(X_train, y_train)
    
    # Logistic Regression

clf = LogisticRegression(max_iter=1000)
clf_model = clf.fit(X_train,y_train)

    # XGBoost
xgb = XGBClassifier()
xgb_model = xgb.fit(X_train, y_train)


# Display Model Performance Metrics

    # Linear Regression

print("Linear Regression Model")
lin_score = lin_model.score(X_train, y_train)
lin_val_score = lin_model.score(X_val, y_val)
print("Train - Mean Accuracy: {0:.4f}".format(lin_score))
print("Validation - Mean Accuracy: {0:.4f}".format(lin_val_score))

    # Logistic Regression

print("Logistic Regression Model")
clf_score = clf_model.score(X_train, y_train)
clf_val_score = clf_model.score(X_val, y_val)
print("Train - Mean Accuracy: {0:.4f}".format(clf_score))
print("Validation - Mean Accuracy: {0:.4f}".format(clf_val_score))

    # XGBoost

print("XGBoost Classification Model")
xgb_score = xgb_model.score(X_train, y_train)
xgb_val_score = xgb_model.score(X_val, y_val)
print("Train - Mean Accuracy: {0:.4f}".format(xgb_score))
print("Validation - Mean Accuracy: {0:.4f}".format(xgb_val_score))

#Prepare the test data in the same manner as the train data.

df_test['Cabin'] = df_test['Cabin'].str.get(0)
df_test['Cabin'] = df_test['Cabin'] + "_cabin"

test_sex_dummies = pd.get_dummies(df_test['Sex'])
test_emb_dummies = pd.get_dummies(df_test['Embarked'])
test_cabin_dummies = pd.get_dummies(df_test['Cabin'])

X_test = pd.concat([df_test, test_sex_dummies, test_emb_dummies,
                    test_cabin_dummies],axis = 1)
X_test = X_test.drop(['Sex','Cabin','Embarked','Name',
                          'Ticket'], axis=1)

X_test = X_test.dropna()

    # Extract ID's for output

X_test_id = X_test['PassengerId']
X_test = X_test.drop(columns='PassengerId')

# Create predictions for X_test data from the models earlier.

lin_y_pred = lin_model.predict(X_test)
log_y_pred = clf_model.predict(X_test)
xgb_y_pred = xgb_model.predict(X_test)

# See if the predicted survival rates are in line with the training data.

    # Linear Regression
print("Linear Regression Model")
print("Test - predicted survival rate: {0:.4f}".format(lin_y_pred.mean()))

print("Logistic Regression Model")
print("Test - predicted survival rate: {0:.4f}".format(log_y_pred.mean()))

print("XGBoost Model")
print("Test - predicted survival rate: {0:.4f}".format(xgb_y_pred.mean()))

# Output our chosen prediction array alongside passenger ID's

out = pd.DataFrame(X_test_id, xgb_y_pred, columns =['PassengerId','Survived'])


=======
# Kaggle - Titanic Dataset Competition

# Imports

import pandas as pd

from pathlib import Path

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

# Set Path

path = Path('C:/Users/hbinn/Documents/Projects/kaggle/titanic/')

# Load Data

df_train = pd.read_csv(path/'train.csv')
df_test = pd.read_csv(path/'test.csv')

# Data Analysis & Exploration **TBA**

tot_surv_rate = df_train['Survived'].mean()

print("The total survival rate of the Titanic sample is %.4f" %tot_surv_rate)

# Feature Engineering and Transfomation

    # Creating Dummy variables for Categorical Data
train_sex_dummies = pd.get_dummies(df_train['Sex'])
train_emb_dummies = pd.get_dummies(df_train['Embarked'])

    # Data Imputation for Missing Variables **TBA**

X = pd.concat([df_train, train_sex_dummies, train_emb_dummies], axis=1)

    # Data Scaling **TBA**

    # Removing unused variables and finalising data before fitting to model.

X = X.drop(['Sex','Cabin','Embarked','Name','PassengerId',
                          'Ticket'], axis=1)

X = X.dropna()
y = X['Survived']
X = X.drop(columns ='Survived')

# Create Train & Validation Data Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)

# Prepare and fit the models we will assess.
    
    # Linear Regression
lin_reg = LinearRegression()
lin_model = lin_reg.fit(X_train, y_train)
    
    # Logistic Regression

clf = LogisticRegression(max_iter=1000)
clf_model = clf.fit(X_train,y_train)

    # XGBoost
xgb = XGBClassifier()
xgb_model = xgb.fit(X_train, y_train)


# Display Model Performance Metrics

    # Linear Regression

print("Linear Regression Model")
lin_score = lin_model.score(X_train, y_train)
lin_val_score = lin_model.score(X_val, y_val)
print("Train - Mean Accuracy: {0:.4f}".format(lin_score))
print("Validation - Mean Accuracy: {0:.4f}".format(lin_val_score))

    # Logistic Regression

print("Logistic Regression Model")
clf_score = clf_model.score(X_train, y_train)
clf_val_score = clf_model.score(X_val, y_val)
print("Train - Mean Accuracy: {0:.4f}".format(clf_score))
print("Validation - Mean Accuracy: {0:.4f}".format(clf_val_score))

    # XGBoost

print("XGBoost Classification Model")
xgb_score = xgb_model.score(X_train, y_train)
xgb_val_score = xgb_model.score(X_val, y_val)
print("Train - Mean Accuracy: {0:.4f}".format(xgb_score))
print("Validation - Mean Accuracy: {0:.4f}".format(xgb_val_score))

#Prepare the test data in the same manner as the train data.

test_sex_dummies = pd.get_dummies(df_test['Sex'])
test_emb_dummies = pd.get_dummies(df_test['Embarked'])

X_test = pd.concat([df_test, test_sex_dummies, test_emb_dummies],axis = 1)
X_test = X_test.drop(['Sex','Cabin','Embarked','Name',
                          'Ticket'], axis=1)

X_test = X_test.dropna()

    # Extract ID's for output

X_test_id = X_test['PassengerId']
X_test = X_test.drop(columns='PassengerId')

# Create predictions for X_test data from the models earlier.

lin_y_pred = lin_model.predict(X_test)
log_y_pred = clf_model.predict(X_test)
xgb_y_pred = xgb_model.predict(X_test)

# See if the predicted survival rates are in line with the training data.

    # Linear Regression
print("Linear Regression Model")
print("Test - predicted survival rate: {0:.4f}".format(lin_y_pred.mean()))

print("Logistic Regression Model")
print("Test - predicted survival rate: {0:.4f}".format(log_y_pred.mean()))

print("XGBoost Model")
print("Test - predicted survival rate: {0:.4f}".format(xgb_y_pred.mean()))

# Output our chosen prediction array alongside passenger ID's

out = pd.DataFrame(X_test_id, xgb_y_pred, columns =['PassengerId','Survived'])


>>>>>>> c0f453d22abc5d0d12d113d18792b6f6a246bacc
