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
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
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


#rbf核函数
clf = SVC(kernel='rbf', C=5) 
#打印参数
# {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None,
#  'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 
# 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1,
#  'probability': True, 'random_state': None, 'shrinking': True,
#  'tol': 0.001, 'verbose': False}
# print(clf.get_params())

# hinge loss
# clf = SVC(kernel='linear', C=0.1)

clf.fit(X_train,y_train)

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



