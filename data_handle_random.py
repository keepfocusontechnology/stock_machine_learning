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
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve


# names=['OPEN-CLOSE', '0PEN-EXCLOSE', 'HIGH-LOW', 'CLOSE-LOW','CLOSE-HIGH','Y']

_mcsv = pd.read_csv("20201126_v2.csv")
data = _mcsv.values[:,:]
y = data[:,5]
X = data[:,:5]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=None)


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
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))

### https://scikit-learn.org/stable/modules/cross_validation.html
scores = cross_val_score(clf, X_train, y_train, cv=5)


def getRandomModel():
    clf = RandomForestClassifier(max_depth=5, n_estimators=25)
    return clf

def plot_learning_curve(algo, X_train, X_test, y_train, y_test):
    """绘制学习曲线：只需要传入算法(或实例对象)、X_train、X_test、y_train、y_test"""
    """当使用该函数时传入算法，该算法的变量要进行实例化，如：PolynomialRegression(degree=2)，变量 degree 要进行实例化"""
    train_score = []
    test_score = []
    for i in range(1, len(X_train)+1):
        algo.fit(X_train[:i], y_train[:i])
 
        y_train_predict = algo.predict(X_train[:i])
        train_score.append(mean_squared_error(y_train[:i], y_train_predict))
    
        y_test_predict = algo.predict(X_test)
        test_score.append(mean_squared_error(y_test, y_test_predict))
    
    plt.plot([i for i in range(1, len(X_train)+1)],
            np.sqrt(train_score), label="train")
    plt.plot([i for i in range(1, len(X_train)+1)],
            np.sqrt(test_score), label="test")
    
    plt.legend()
    plt.axis([0, len(X_train)+1, 0, 4])
    plt.show()


def learnCurve(algo):
    train_sizes,train_loss,test_loss=learning_curve(
        algo,X,y,cv=5,scoring='neg_mean_squared_error',
        train_sizes=[0.1,0.25,0.5,0.75,1])
    train_loss_mean = -np.mean(train_loss,axis=1)    
    test_loss_mean = -np.mean(test_loss,axis=1)

    plt.plot(train_sizes,train_loss_mean,'o-',color="r",label="Training")
    plt.plot(train_sizes,test_loss_mean,'o-',color="y",label="Cross-validation")
    plt.xlabel("Training examples")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()


# plot_learning_curve(getRandomModel(),X_train, X_test, y_train, y_test)
learnCurve(getRandomModel())
# print(scores)