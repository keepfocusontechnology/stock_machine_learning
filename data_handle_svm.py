import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif']=[u'simHei']
matplotlib.rcParams['axes.unicode_minus']=False
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# names=['OPEN-CLOSE', 'OPEN-EXCLOSE', 'HIGH-LOW', 'CLOSE-LOW','PRE-V','Y']

# OPEN-CLOSE= 当日收盘比开盘的涨跌幅
# OPEN-EXCLOSE = 当日开盘比昨日收盘的涨跌幅
# HIGH-LOW = 当日收盘比当日最低高的幅度
# CLOSE-LOW = 当日收盘比当日最高低的幅度
# PRE-V成交量涨跌幅

_mcsv = pd.read_csv("20201126_v2.csv")
data = _mcsv.values[:,:]


X = data[:,:5]
y = data[:,5]

# plt.scatter(X[:,1],X[:,3],c=y, alpha=0.5, cmap='viridis')
# plt.colorbar()  # 显示颜色条
# plt.show()


# print(X)
# print(y)
# print(data)

# print('数据X总数： {}'.format(X.size))
# print('数据y总数： {}'.format(y.size))

# X_test,y_test数据占总数据集的百分比
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3)


def findRandomState():
    i = 0
    while i <= 22:
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=i)
        print("state=",i)
        normalPredict(X_train,X_test,y_train,y_test,5,12.5)
        i=i+1



def showCAccuracy():
    c_range = range(1,500)
    c_scores=[]
    for c in c_range:
        clf = SVC(kernel='rbf', C=c)
        scores = cross_val_score(clf,X,y,cv=10,scoring='accuracy')
        # 误差 for regression
        # loss = cross_val_score(clf,X,y,cv=10,scoring='r2')
        c_scores.append(scores.mean())

    plt.plot(c_range,c_scores)
    plt.xlabel('Value of C for SVM')
    plt.ylabel('Cross-Validated Accuracy')    
    plt.show()




def showGammaAccuracy():
   
    g_range=range(1,20)
    g_scores=[]
    for g in g_range:
        clf = SVC(kernel='rbf', gamma=g)
        scores = cross_val_score(clf,X,y,cv=10,scoring='accuracy')
        g_scores.append(scores.mean())

    plt.plot(g_range,g_scores)
    plt.xlabel('Value of C for SVM')
    plt.ylabel('Cross-Validated Accuracy')    
    plt.show()





def normalPredict(X_train,X_test,y_train,y_test,c_value,g_value):

    clf = getModel(X_train,X_test,y_train,y_test,c_value,g_value)
    
    #打印参数
    # {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None,
    #  'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 
    # 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1,
    #  'probability': True, 'random_state': None, 'shrinking': True,
    #  'tol': 0.001, 'verbose': False}
    # print(clf.get_params())

    # hinge loss
    # clf = SVC(kernel='linear', C=0.1)
    # # 线性函数可用，x变量的权重
    # print(clf.coef_)
    # # 截距
    # print(clf.intercept_)
    # print(clf.predict(X_test))
    # print(y_test)
    predictions = clf.predict(X_test)
    # precision recall f1-score三列分别为各个类别的精确度/召回率及 F1值
    # 精确率是针对我们预测结果而言的，它表示的是预测为正的样本中有多少是真正的正样本。
    # 那么预测为正就有两种可能了，一种就是把正类预测为正类(TP)，另一种就是把负类预测为正类(FP)，
    # 也就是而召回率是针对我们原来的样本而言的，它表示的是样本中的正例有多少被预测正确了。
    # 那也有两种可能，一种是把原来的正类预测成正类(TP)，另一种就是把原来的正类预测为负类(FN)。

    #  F1值
    #  F1值是精确度和召回率的调和平均值：
    # 精确度和召回率都高时， F1值也会高． F1值在1时达到最佳值（完美的精确度和召回率），最差为0．在二元分类中， F1值是测试准确度的量度。

    print(classification_report(y_test,predictions))
    # 匹配度,利用R^2评分 coefficient of determination
    # print(clf.score(X_test,y_test)) 
    print("AC",accuracy_score(y_test,predictions))
    scores = cross_val_score(clf,X,y,cv=5,scoring='accuracy')
    #5折交叉验证平均值
    print(scores.mean())



def getModel(X_train,X_test,y_train,y_test,c_value,g_value):
    #rbf核函数
    clf = SVC(kernel='rbf', C=c_value,gamma=g_value) 
    clf.fit(X_train,y_train)
    return clf

def defaultModel():
    clf = SVC(kernel='rbf') 
    clf.fit(X_train,y_train)
    # print(clf.score(X_test,y_test))
    # scores_test = cross_val_score(clf,X_test,y_test,cv=5,scoring='accuracy') 
    scores = cross_val_score(clf,X,y,cv=5,scoring='accuracy')    
    print(scores)
    # print('test',scores_test)
    print(scores.mean())
    # print('test mean',scores_test.mean())





def learnCurve():
    train_sizes,train_loss,test_loss=learning_curve(
        SVC(),X,y,cv=5,scoring='neg_mean_squared_error',
        train_sizes=[0.1,0.25,0.5,0.75,1])
    train_loss_mean = -np.mean(train_loss,axis=1)    
    test_loss_mean = -np.mean(test_loss,axis=1)

    plt.plot(train_sizes,train_loss_mean,'o-',color="r",label="Training")
    plt.plot(train_sizes,test_loss_mean,'o-',color="y",label="Cross-validation")
    plt.xlabel("Training examples")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()

# validation_curve 要看的是 SVC() 的超参数 gamma，
# gamma 的范围是取 0.1到 10, 取10个点，
# 评分用的是 neg_mean_squared_error
def validationCurveGamma():
    param_range = np.logspace(0, 2, 5)
    train_loss,test_loss=validation_curve(
        SVC(),X,y,param_name='gamma',param_range=param_range,cv=10,scoring='neg_mean_squared_error')
    train_loss_mean = -np.mean(train_loss,axis=1)    
    test_loss_mean = -np.mean(test_loss,axis=1)
    plt.plot(param_range,train_loss_mean,'o-',color="r",label="Training")
    plt.plot(param_range,test_loss_mean,'o-',color="y",label="Cross-validation")
    plt.xlabel("gamma")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()


# validation_curve 要看的是 SVC() 的超参数 C
# c 的范围是取 1到 30, 取30个点，
# 评分用的是 neg_mean_squared_error
def validationCurveC():
    param_range = range(1,30)
    train_loss,test_loss=validation_curve(
        SVC(),X,y,param_name='C',param_range=param_range,cv=10,scoring='accuracy')
    train_loss_mean = np.mean(train_loss,axis=1)    
    test_loss_mean = np.mean(test_loss,axis=1)
    plt.plot(param_range,train_loss_mean,'o-',color="r",label="Training")
    plt.plot(param_range,test_loss_mean,'o-',color="y",label="Cross-validation")
    plt.xlabel("C")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()

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





# 验证C的取值范围  c =8
# showCAccuracy() 
# 验证Gamma取值范围 gmama = 2
# showGammaAccuracy()
# 输出准确度,5折交叉验证
normalPredict(X_train,X_test,y_train,y_test,20,31)
# 输出模型学习程度
# learnCurve()
# 验证gamma函数 => gamma = 0.1
# validationCurveGamma()
#  验证C的值 => C=8
# validationCurveC()
# plot_learning_curve(SVC(C=8,gamma=2.12),X_train, X_test, y_train, y_test)

# defaultModel()

