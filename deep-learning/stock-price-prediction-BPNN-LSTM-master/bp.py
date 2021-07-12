import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns', None)



def open_excel(path,sheet_name_str):
    df = pd.read_excel (path, sheet_name=sheet_name_str)
    return df





#读取数据，0对应第一支股票，1对应第二只，以此类推
# df1=open_excel('/Users/liuzhibo/Downloads/stock-price-prediction-BPNN-LSTM-master/data.xlsx',0)
# df1=df1.iloc[3600:-10,1:] # 选取从第x行开始的数据 到 y 截止 ， 第二列开始 就是 时间不要，留10个做预测
df1 = open_excel('/Users/mr_qual1ty/python_pro/deep-learning/stock-price-prediction-BPNN-LSTM-master/data_bp.xlsx','001')
df1 = df1.iloc[:-10,1:]

# print(df1.tail())





# 归一化过程
print("------------------------------数据归一开始------------------------------")
min_max_scaler = preprocessing.MinMaxScaler()
df0=min_max_scaler.fit_transform(df1)
df = pd.DataFrame(df0, columns=df1.columns)
X=df.iloc[:,:-1]
y=df['实收金额']     #切片是前闭后开[)    这里面 target 就是 预测目标
print('x.shape = ',X.shape)
print('y.shape = ',y.shape)




y=pd.DataFrame(y.values,columns=['goal'])
x=X
cut=20#取最后cut=10天为测试集
X_train, X_test=x.iloc[:-cut],x.iloc[-cut:]#列表的切片操作，X.iloc[0:2400，0:7]即为1-2400行，1-7列
y_train, y_test=y.iloc[:-cut],y.iloc[-cut:]
X_train,X_test,y_train,y_test=X_train.values,X_test.values,y_train.values,y_test.values
print('X_train.size = f%',X_train.size)#通过输出训练集测试集的大小来判断数据格式正确。
print(X_test.size)
print(y_train.size)
print(y_test.size)

print("------------------------------数据归一结束-----------------------------")

print("------------------------------模型建立-----------------------------")

model = Sequential()  #层次模型
model.add(Dense(16,input_dim=68)) #输入层，Dense表示BP层   init = 是个什么鬼    这里曾经崩溃过一次，应该是参数不对
model.add(Activation('relu'))  #添加激活函数
model.add(Dense(4)) #中间层
model.add(Activation('sigmoid'))  #添加激活函数
model.add(Dense(1))  #输出层
model.compile(loss='mean_squared_error', optimizer='Adam') #编译模型
model.fit(X_train, y_train, epochs = 200, batch_size = 256) #训练模型nb_epoch=50次 变成100，200，300，400 开始变好，但不知道为什么300不好了，但好的结果我也发现了问题了   nb_epoch又是什么鬼

model.summary()#模型描述

y_train_predict=model.predict(X_train)
y_train_predict=y_train_predict[:,0]
y_train=y_train

draw=pd.concat([pd.DataFrame(y_train),pd.DataFrame(y_train_predict)],axis=1)
draw.iloc[100:400,0].plot(figsize=(12,6))
draw.iloc[100:400,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),fontsize='15')
plt.title("Train Data",fontsize='30') #添加标题
plt.show()
#展示在训练集上的表现

y_test_predict=model.predict(X_test)
y_test_predict=y_test_predict[:,0]


draw=pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_test_predict)],axis=1);
draw.iloc[:,0].plot(figsize=(12,6))
draw.iloc[:,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Test Data",fontsize='30') #添加标题
plt.show()
#展示在测试集上的表现



print('训练集上的MAE/MSE/MAPE')
print(mean_absolute_error(y_train_predict, y_train))
print(mean_squared_error(y_train_predict, y_train) )
print(mape(y_train_predict, y_train) )
print('测试集上的MAE/MSE/MAPE')
print(mean_absolute_error(y_test_predict, y_test))
print(mean_squared_error(y_test_predict, y_test) )
print(mape(y_test_predict, y_test) )
y_var_test=y_test[1:]-y_test[:len(y_test)-1]
y_var_predict=y_test_predict[1:]-y_test_predict[:len(y_test_predict)-1]
txt=np.zeros(len(y_var_test))
for i in range(len(y_var_test-1)):
    txt[i]=np.sign(y_var_test[i])==np.sign(y_var_predict[i])
result=sum(txt)/len(txt)
print('预测涨跌正确:',result)