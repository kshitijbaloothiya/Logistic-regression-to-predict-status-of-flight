import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn import metrics
from sklearn.feature_selection import RFE
import numpy as np
from sklearn.feature_selection import SelectFromModel
flight = pd.read_csv("C:/Users/91704/Documents/IIT/SEM-2/Machine learning/assignment-rw/FlightDelays.csv")
day_m = flight['DAY_OF_MONTH'].tolist()  # DAY OF MONTH
day_w = flight['DAY_WEEK'].tolist()      # DAY OF WEEK
status = flight['Flight Status']
wea = flight['Weather'].tolist()         # WEATHER CONDITION
carr = flight['CARRIER'].tolist()        # CARRIER INFORMATION
origin = flight['ORIGIN'].tolist()       # ORIGIN AIRPORT
desti = flight['DEST'].tolist()          # DESTINATION AIRPORT
si1 = []
si2 = []
si3 = []
si4 = []
for i in range(0,len(status)):
    if status[i] == 'delayed' and wea[i] == 0:  # NOT INCLUDING THE DATAS FOR WHICH DELAY IS DUE TO WHEATHER
        si1.append(day_w[i])
        si2.append(carr[i])
        si3.append(origin[i])
        si4.append(desti[i])
# DELAY ON A PARTICULAR DAY
labels = 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
sizes = [si1.count(1),si1.count(2), si1.count(3), si1.count(4), si1.count(5), si1.count(6), si1.count(7)]
plt.plot(sizes)
plt.xlabel(labels)
plt.show()
# DELAY OF A PARTICULAR CARRIER
labels1 = 'CO', 'DH', 'OH', 'DL', 'MQ', 'RU', 'UA', 'US'
sizes1 = [si2.count('CO'), si2.count('DH'), si2.count('OH'), si2.count('DL'), si2.count('MQ'), si2.count('RU'), si2.count('UA'), si2.count('US')]
plt.plot(sizes1)
plt.xlabel(labels1)
plt.show()
# DELAY OF ORIGIN FLIGHT
labels2 = 'DCA', 'IAD', 'BWI'
sizes2 = [si3.count('DCA'), si3.count('IAD'), si3.count('BWI')]
plt.plot(sizes2)
plt.xlabel(labels2)
plt.show()
# DELAY ON DESTINATION SIDE
labels3 = 'JFK', 'LGA', 'EWR'
sizes3 = [si4.count('JFK'), si4.count('LGA'), si4.count('EWR')]
plt.plot(sizes3)
plt.xlabel(labels3)
plt.show()
#Q.2
carr = pd.get_dummies(carr)
origin = pd.get_dummies(origin)
desti = pd.get_dummies(desti)
day_m = pd.get_dummies(day_m)
day_w = pd.get_dummies(day_w)
X = pd.concat([carr, origin, desti, day_m, day_w], axis=1, sort=False) # WEATHER IS NOT INCLUDED
y = []
r1 = len(status)
for i in range(0,r1):
    if status[i] == 'ontime':
        y.append(1)
    else:
        y.append(0)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4)
X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train)
X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)
LR = LogisticRegression()
LR.fit(X_train, y_train)
yhat = LR.predict(X_test)
print(metrics.accuracy_score(y_test, yhat))
# Q3
coeff = LR.coef_
coeff = coeff[0]
# Q4&5
coeff = coeff.tolist()
# no of features
nof_list=np.arange(1,len(coeff))
high_score=0
# Variable to store the optimum features using recursive feature elimination
nof=0
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4)
    X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train)
    X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)
    mdl = LogisticRegression()
    rfe = RFE(mdl,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    mdl.fit(X_train_rfe,y_train)
    score = mdl.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
# Feature selection using lasso
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4)
sel = SelectFromModel(LogisticRegression())
sel.fit(X_train, y_train)
selected_feat = X_train.columns[(sel.get_support())]
print(len(selected_feat))
X_train_lasso = sel.fit_transform(X_train, y_train)
X_test_lasso = sel.transform(X_test)
mdl_lasso = LogisticRegression()
mdl_lasso.fit(X_train_lasso, y_train)
score_lasso = mdl_lasso.score(X_test_lasso, y_test)
print(score_lasso)