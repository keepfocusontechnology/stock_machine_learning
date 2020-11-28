import pandas as pd
import matplotlib
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
from sklearn.model_selection import learning_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

def draw_ROC_curve(y_test,y_predict):
    false_positive_rate,true_positive_rate,thresholds=roc_curve(y_test, y_predict)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()

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

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=None)

# Similar to SVC with parameter kernel=’linear’, but implemented
# in terms of liblinear rather than libsvm, so it has more flexibility
# in the choice of penalties and loss functions and should scale better
# to large numbers of samples.
clf = Pipeline([
    ('standardscaler', StandardScaler()),
    ('linearsvc', LinearSVC(C=1.0, random_state=0, tol=1e-05))
])
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("LinearSVC")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))

draw_ROC_curve(y_test, predictions)

### https://scikit-learn.org/stable/modules/cross_validation.html
scores = cross_val_score(clf, X_train, y_train, cv=5)
print(scores)

## how a classifier is optimized by cross-validation, which is done using the
# sklearn.model_selection.GridSearchCV object on a development set 
# that comprises only half of the available labeled data.
tuned_parameters = [
    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 2, 3]},
    {'kernel': ['linear'], 'C': [1, 2, 3]}
]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, cv=5
    )
    clf.fit(X_train, y_train)
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()