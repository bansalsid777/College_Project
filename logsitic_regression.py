
"""Logistic Regression Model Used For Classification of Binary Values"""
"""DATASET Used here is KC1  provided by NASA for classification model"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('KC1.csv')

X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, 21].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
"""spliting the dataset into 20:80 ratio 20 for testing and 80 for training the model"""
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
betaHat=classifier.coef_                           # Xt.X.BetaHAt=Xt.Y
print(betaHat)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
"""It shows comparison between Actual And predicted values """
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

precision=cm[1,1]/(cm[1,0] + cm[1,1])

speci= cm[0,1]/(cm[0,1]+cm[1,1])

PD=cm[1,1]/(cm[0,1]+cm[1,1])
Pf=cm[1,0,]/(cm[0,0]+cm[1,0])

from sklearn.model_selection import cross_val_score
acc=cross_val_score(estimator=classifier, X=X_train,y=y_train,cv=10)
ACC=acc.mean()
print(ACC)
acc.std() 

#Root mean Squared value
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, y_pred))

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))


"""ROC(reciever operating characterstics)"""
"""tpr=true positive rate"""
"""fpr= flase positive rate"""

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_pred,y_test)
roc_auc = auc(fpr, tpr)

G1m=(PD*precision)**0.5
G2m=(PD*speci)**0.5

F_mean=PD-Pf

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import f1_score
f1score=f1_score(y_test, y_pred,average='micro') 
print(f1score)
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, _ = precision_recall_curve(y_test, y_pred)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
