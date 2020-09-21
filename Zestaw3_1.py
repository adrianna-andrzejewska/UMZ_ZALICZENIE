import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

# copied from shared materials
try:
    from sklearn.metrics import plot_confusion_matrix
except(ImportError):
    from sklearn.utils.multiclass import unique_labels

    def plot_confusion_matrix(y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.xlim(-0.5, len(classes)-0.5)
        plt.ylim(len(classes)-0.5, -0.5)
        return ax
# loading data
data_file = os.path.join('data', 'train.tsv')
test_data_file = os.path.join('data', 'test.tsv')
results_file = os.path.join('data', 'results.tsv')
output_file = os.path.join('data', 'out.tsv')

# add names to columns and delete na
df_names = ['Occupancy',
            'Date',
            'Temperature',
            'Humidity',
            'Light',
            'CO2',
            'HumidityRatio']
df = pd.read_csv(data_file, sep='\t', names=df_names)
df = df.dropna()
df.describe()

occupancy_percentage = sum(df.Occupancy) / len(df)
print("Occupancy percentage is: " + str(occupancy_percentage))
print("Zero rule model accuracy on training set is: " +
      str(1 - occupancy_percentage))
clf = LogisticRegression()
X_train = df[['Temperature']]
y_train = df.Occupancy

# train and prediction
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)

# accuracy
clf_accuracy = sum(y_train == y_train_pred) / len(df)
print("Training set accuracy for logisitic regression model " +
      "on Temperature variable:\n" + str(clf_accuracy))

accuracy_score(y_train, y_train_pred)

# confusion_matrix
conf_matrix = confusion_matrix(y_train, y_train_pred)
tn, fp, fn, tp = conf_matrix.ravel()

sensitivity = conf_matrix[0, 0]/(conf_matrix[0, 0]+conf_matrix[0, 1])
print('Sensitivity : ', sensitivity)

specificity = conf_matrix[1, 1]/(conf_matrix[1, 0]+conf_matrix[1, 1])
print('Specificity : ', specificity)

df_column_names = ['Date', 'Temperature', 'Humidity', 'Light',
                   'CO2', 'HumidityRatio']
X_column_names = ['Temperature',
                  'Humidity',
                  'Light',
                  'CO2',
                  'HumidityRatio']

X_test = pd.read_csv(test_data_file,
                     sep='\t',
                     names=df_column_names,
                     usecols=X_column_names)

# test set
df_results = pd.read_csv(results_file, sep='\t', names=['y'])
df_results['y'] = df_results['y'].astype('category')

y_true = df_results['y']

# test accuracy

y_test_pred = clf.predict(X_test[['Temperature']])
clf_test_accuracy = accuracy_score(y_true, y_test_pred)
print('Accuracy on test dataset (full model): ' + str(clf_test_accuracy))

# logistic regression classifier on all but date independent variables
clf_all = LogisticRegression()
X_train_all = df[['Temperature',
                  'Humidity',
                  'Light',
                  'CO2',
                  'HumidityRatio']]
y_train_pred_all = clf_all.predict(X_train_all)

# accuracy
clf_all_accuracy = accuracy_score(y_train, y_train_pred_all)
print("Training set accuracy for logisitic regression model " +
      "on all but date variable: " + str(clf_all_accuracy))
clf_all.fit(X_train_all, y_train)

conf_matrix = confusion_matrix(y_train, y_train_pred_all)
tn, fp, fn, tp = conf_matrix.ravel()

sensitivity = conf_matrix[0, 0]/(conf_matrix[0, 0]+conf_matrix[0, 1])
print('Sensitivity : ', sensitivity)

specificity = conf_matrix[1, 1]/(conf_matrix[1, 0]+conf_matrix[1, 1])
print('Specificity : ', specificity)

y_test_pred_all = clf_all.predict(X_test)
clf_test_accuracy = accuracy_score(y_true,
                                   y_test_pred_all)
print('Accuracy on test dataset (full model): ' + str(clf_test_accuracy))

# save to csv
pd.Series(data=y_test_pred).to_csv(output_file, index=False)
