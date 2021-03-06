# 

<u>第一小组股票学习</u>



## 使用SVM算法确定预测股票涨跌情况

引子：对应一只股票，我们一般只希望预测它接下来的走势是涨是跌，属于分类问题。如果把股票数据集在一个平面表示出来，我们希望找到一条最合适的线将其划分为两个部分。而这恰恰是**SVM（Support Vector Machines）**最擅长的事儿。

理想情况：

![101606666273_.pic](/Users/quat1ly/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/f1f46661f5043a6dcb869bad9a5cc562/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/101606666273_.pic.jpg)





#### 一. SVM介绍

**1.为什么要研究线性分类？**

在讨论SVM之前首先研究，为什么要把数据集用**线性分类**分开？难道不可以用非线性分开吗？

首先，非线性分开的复杂度非常之大，而线性分开只需要一个平面。其次，非线性分开仅就二维空间而言，曲线，折线，双曲线，波浪线，以及毫无规律的各种曲线太多，没有办法进行统一处理。即使针对某一个具体问题处理得到了非线性分类结果，也无法很好的推广到其他情形，用非线性处理分类问题，耗时耗力。

**2.SVM（Support Vector Machines）思想是什么**

2.1 硬间隔支持向量机

SVM中最关键的思想之一就是引入和定义了“间隔”这个概念。这个概念本身很简单，以二维空间为例，就是点到分类直线之间的距离。假设直线为y=wx+b，那么只要使所有正分类点到该直线的距离与所有负分类点到该直线的距离的总和达到最大，这条直线就是最优分类直线。这样，原问题就转化为一个约束优化问题，可以直接求解。这叫做硬间隔最大化，得到的SVM模型称作**硬间隔支持向量机**。

 2.2 软间隔支持向量机

但是新问题出现了，在实际应用中，我们得到的数据并不总是完美的线性可分的。



<img src="https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Support%20vector%20machine/output_04.png" alt="机器学习| 支持向量机详解(Python 语言描述) - Laugh's blog" style="zoom:80%;" />

其中可能会有个别噪声点，他们错误的被分类到了其他类中。如果将这些特异的噪点去除后，可以很容易的线性可分。但是，我们对于数据集中哪些是噪声点却是不知道的，如果以之前的方法进行求解，会无法进行线性分开。是不是就没办法了呢？假设在y=x+1直线上下分为两类，若两类中各有对方的几个噪点，在人的眼中，仍然是可以将两类分开的。这是因为在人脑中是可以容忍一定的误差的，仍然使用y=x+1直线分类，可以在最小误差的情况下进行最优的分类。同样的道理，我们在SVM中引入误差的概念，将其称作“**松弛变量**”。通过加入松弛变量，在原距离函数中需要加入新的松弛变量带来的误差，这样，最终的优化目标函数变成了两个部分组成：距离函数和松弛变量误差。这两个部分的重要程度并不是相等的，而是需要依据具体问题而定的，因此，我们加入权重参数C，将其与目标函数中的松弛变量误差相乘，这样，就可以通过调整C来对二者的系数进行调和。如果我们能够容忍噪声，那就把C调小，让他的权重降下来，从而变得不重要；反之，我们需要很严格的噪声小的模型，则将C调大一点，权重提升上去，变得更加重要。通过对参数C的调整，可以对模型进行控制。这叫做软间隔最大化，得到的SVM称作**软间隔支持向量机**。

 2.3 非线性支持向量机

 之前的硬间隔支持向量机和软间隔支持向量机都是解决线性可分数据集或近似线性可分数据集的问题的。但是如果噪点很多，甚至会造成数据变成了线性不可分的，那该怎么办？最常见的例子是在二维平面笛卡尔坐标系下，以原点(0,0)为圆心，以1为半径画圆，则圆内的点和圆外的点在二维空间中是肯定无法线性分开的。但是，学过初中几何就知道，对于圆圈内（含圆圈）的点：x^2+y^2≤1，圆圈外的则x^2+y^2＞1。我们假设第三个维度：z=x^2+y^2，那么在第三维空间中，可以通过z是否大于1来判断该点是否在圆内还是圆外。这样，在二维空间中线性不可分的数据在第三维空间很容易的线性可分了。这就是**非线性支持向量机**。

这是SVM非常重要的思想。对于在N维空间中线性不可分的数据，在N+1维以上的空间会有更大到可能变成线性可分的（但并不是一定会在N+1维上线性可分。维度越高，线性可分的可能性越大，但并不完全确保）。因此，对于线性不可分的数据，我们可以将它映射到线性可分的新空间中，之后就可以用刚才说过的硬间隔支持向量机或软间隔支持向量机来进行求解了。这样，我们将原问题变成了如何对原始数据进行映射，才能使其在新空间中线性可分。在上面的例子中，通过观察可以使用圆的方程来进行映射，但在实际数据中肯定没有这么简单。如果都可以观察出规律来，那就不需要机器来做SVM了。。

实际中，对某个实际问题函数来寻找一个合适的空间进行映射是非常困难的，幸运的是，在计算中发现，我们需要的只是两个向量在新的映射空间中的内积结果，而映射函数到底是怎么样的其实并不需要知道。这一点不太好理解，有人会问，既然不知道映射函数，那怎么能知道映射后在新空间中的内积结果呢？答案其实是可以的。这就需要引入了核函数的概念。核函数是这样的一种函数：仍然以二维空间为例，假设对于变量x和y，将其映射到新空间的映射函数为φ，则在新空间中，二者分别对应φ(x)和φ(y)，他们的内积则为<φ(x),φ(y)>。我们令函数Kernel(x,y)=<φ(x),φ(y)>=k(x,y)，可以看出，函数Kernel(x,y)是一个关于x和y的函数！而与φ无关！这是一个多么好的性质！我们再也不用管φ具体是什么映射关系了，只需要最后计算Kernel(x,y)就可以得到他们在高维空间中的内积，这样就可以直接带入之前的支持向量机中计算！真是妈妈再也不用担心我的学习了。。

得到这个令人欢欣鼓舞的函数之后，我们还需要冷静一下，问问：这个Kernel函数从哪来？他又是怎么得到的？真的可以解决所有映射到高维空间的问题吗？

这个问题我试着回答一下，如果我理解对的话。核函数不是很好找到，一般是由数学家反向推导出来或拼凑出来的。现在知道的有多项式核函数、高斯核函数、字符串核函数等。其中，高斯核函数对应的支持向量机是高斯径向基函数（RBF），是最常用的核函数。

RBF核函数可以将维度扩展到无穷维的空间，因此，理论上讲可以满足一切映射的需求。为什么会是无穷维呢？我以前都不太明白这一点。后来老师讲到，RBF对应的是泰勒级数展开，在泰勒级数中，一个函数可以分解为无穷多个项的加和，其中，每一个项可以看做是对应的一个维度，这样，原函数就可以看做是映射到了无穷维的空间中。这样，在实际应用中，RBF是相对最好的一个选择。当然，如果有研究的话，还可以选用其他核函数，可能会在某些问题上表现更好。但是，RBF是在对问题不了解的情况下，对最广泛问题效果都很不错的核函数。因此，使用范围也最广。

这样，对于线性不可分的数据，也可以通过RBF等核函数来映射到高维，甚至无穷维的空间中而变得线性可分，通过计算间隔和松弛变量等的最大化，可以对问题进行求解。当然，在求解中，还有一些数学的技巧来简化运算，例如，使用拉格朗日乘子来将原问题变换为对偶问题，可以简化计算。这些在实验中用不到，而且数学原理有点困难，就先不讲了。





#### 二.模型搭建

- 环境配置
- 工具类





#### 三. 代码详解

```python
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
clf = SVC(kernel='rbf', probability=True,C=10) 
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

# 匹配度,利用R^2评分 coefficient of determination

# precision recall f1-score三列分别为各个类别的精确度/召回率及 F1值

# 实际上非常简单，精确率是针对我们预测结果而言的，它表示的是预测为正的样本中有多少是真正的正样本。
# 那么预测为正就有两种可能了，一种就是把正类预测为正类(TP)，另一种就是把负类预测为正类(FP)，
# 也就是而召回率是针对我们原来的样本而言的，它表示的是样本中的正例有多少被预测正确了。
# 那也有两种可能，一种是把原来的正类预测成正类(TP)，另一种就是把原来的正类预测为负类(FN)。

#  F1值
#  F1值是精确度和召回率的调和平均值：

#  2F1=1P+1R
#  F1=2P×RP+R
# 精确度和召回率都高时， F1值也会高． F1值在1时达到最佳值（完美的精确度和召回率），最差为0．在二元分类中， F1值是测试准确度的量度。

predictions = clf.predict(X_test)

print(classification_report(y_test,predictions))
# print(clf.score(X_test,y_test)) 
print("AC",accuracy_score(y_test,predictions))




```







#### 四.结果输出 



<img src="/Users/quat1ly/Library/Application Support/typora-user-images/image-20201130003736591.png" alt="image-20201130003736591" style="zoom:50%;" />



#### 五.结果验证

