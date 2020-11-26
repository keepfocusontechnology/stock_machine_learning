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

# names=['OPEN-CLOSE', '0PEN-EXCLOSE', 'HIGH-LOW', 'CLOSE-LOW','CLOSE-HIGH','Y']

_mcsv = pd.read_csv("20201126_v2.csv")
data = _mcsv.values[:,:]
y = data[:,5]
X = data[:,:5]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.9,random_state=None)


from sklearn.ensemble import RandomForestClassifier #随机森林分类模型

# print(X_train)
# print("==============================================")
# print(X_test)
# print("==============================================")
# print(y_train)
# print("==============================================")
# print(y_test)
# print("==============================================")

###
clf = RandomForestClassifier(max_depth=5, n_estimators=25)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("RandomForest kernel:linear")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))

### https://scikit-learn.org/stable/modules/cross_validation.html
scores = cross_val_score(clf, X_train, y_train, cv=5)
print(scores)