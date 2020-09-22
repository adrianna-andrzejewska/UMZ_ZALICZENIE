import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

# loading data, 'respondent' is a row's id
df_data_public = pd.read_csv('survey_results_public.csv',
                             header=0,
                             index_col=['Respondent'])

# delete na
df_data_public['Hobbyist'].dropna(inplace=True)
df_data_public['Age'].dropna(inplace=True)

df_data_public['Hobbyist'] = df_data_public[
                                'Hobbyist'].map({'Yes': 1, 'No': 0})

df_data_public['Age'].round(0)
df_data_public.dropna(inplace=True)

clf = LogisticRegression()
X_train = df_data_public[['Hobbyist']]
y_train = df_data_public['Age']

column_values = y_train.values.ravel()
unique_values = pd.unique(column_values)
print(unique_values)
# train
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)

clf_all_accuracy = accuracy_score(y_train, y_train_pred)
print("Training set accuracy for logisitic regression model " +
      str(clf_all_accuracy))

# divide on train and test
X_train_n,
X_test_n,
y_train_n,
y_test_n = sklearn.model_selection.train_test_split(X_train,
                                                    y_train,
                                                    random_state=np.random)
clf.fit(X_train_n, y_train_n)
y_test_pred = clf.predict(X_test_n)

clf_all_accuracy = accuracy_score(y_test_n, y_test_pred)
print("Training set accuracy for logisitic regression model " +
      str(clf_all_accuracy))
