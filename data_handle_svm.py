import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif']=[u'simHei']
matplotlib.rcParams['axes.unicode_minus']=False
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# names=['OPEN-CLOSE', 'OPEN-EXCLOSE', 'HIGH-LOW', 'CLOSE-LOW','PRE-V','Y']

# OPEN-CLOSE= 当日收盘比开盘的涨跌幅
# OPEN-EXCLOSE = 当日开盘比昨日收盘的涨跌幅
# HIGH-LOW = 当日收盘比当日最低高的幅度
# CLOSE-LOW = 当日收盘比当日最高低的幅度
# PRE-V成交量涨跌幅

_mcsv = pd.read_csv("20201126_v2.csv")
data = _mcsv.values[:,:]
y = data[:,5]
X = data[:,:5]

X_train,X_test,y_train,y_test = train_test_split(X,y)

# print(X_train) 
# print('X_train size： {}'.format(X_train.size))
# print("==============================================")
# print(X_test)
# print('X_test size： {}'.format(X_test.size))
# print("==============================================")
# print(y_train)
# print('y_train size： {}'.format(y_train.size))
# print("==============================================")
# print(y_test)
# print('y_test size： {}'.format(y_test.size))
# print("==============================================")


from sklearn.svm import SVC
clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("SVM kernel:rbf")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))

#
# clf = SVC(kernel='linear', C=1.5)
# clf.fit(X_train,y_train)
# predictions = clf.predict(X_test)
# print("SVM kernel:linear")
# print(classification_report(y_test,predictions))
# print("AC",accuracy_score(y_test,predictions))

# ## Pipeline
# clf = Pipeline([
#     ('scaler', StandardScaler()),
#     ('svm_clf', SVC(kernel='linear', C=1)),
# ])
# clf.fit(X_train,y_train)
# print("Pipeline SVM kernel:linear")
# print(classification_report(y_test,predictions))
# print("AC",accuracy_score(y_test,predictions))

### https://scikit-learn.org/stable/modules/cross_validation.html
scores = cross_val_score(clf, X_train, y_train, cv=5)
print(scores)