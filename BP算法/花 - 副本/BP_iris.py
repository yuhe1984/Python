import matplotlib.pyplot as plt
import numpy as np # 导入numpy并命名为np
from sklearn.datasets import *
from sklearn.preprocessing import LabelBinarizer #标签二值化
from sklearn.model_selection import train_test_split   #切割数据,交叉验证法
#载入数据:8*8的数据集
digits = load_iris()
X = digits.data
Y = digits.target
#sklearn切分数据
YY = LabelBinarizer().fit_transform(Y)
X_train, X_test, y_train, y_test, Y_train, Y_test = train_test_split(X,Y,YY,test_size = 0.5, random_state = 42)
xrow,xcol = X_train.shape
yrow,ycol = Y_train.shape
x = X_train
y = Y_train
#experiment of fixed learning rate
'''from BP_network import *
bpn1 = BP_network()  
bpn1.CreateNN(xcol, 5, ycol, actfun = 'Sigmoid', learningrate = 0.05)
e = []
for i in range(1000): 
    err, err_k = bpn1.TrainStandard(x, y)
    e.append(err)
#bpn1.printf(x)
print('固定学习率的误差',err)
 
f1 = plt.figure(1) 
plt.xlabel("epochs")
plt.ylabel("error")
plt.ylim(0,1) 
plt.title("training error convergence curve with fixed learning rate")
plt.plot(e)

#plt.show()

# get the test error in test set
pred = bpn1.PredLabel(X_test);
count  = 0
for i in range(len(y_test)) :
    if pred[i] == y_test[i]: count += 1

zhengquelv = count/len(y_test)
print("正确率: %.6f" % zhengquelv)

print('='*80)'''

#experiment of dynamic learning rate
from BP_networkk import *
bpn2 = BP_networkk()  
bpn2.CreateNN(xcol, 5, ycol, actfun = 'Sigmoid', learningrate = 0.1)
e = []
for i in range(1000): 
    err, err_k = bpn2.TrainStandard(x, y)
    e.append(err)
#bpn2.printf(x)
print('动态学习率的误差',err)
 
f2 = plt.figure(2) 
plt.xlabel("epochs")
plt.ylabel("error")
plt.ylim(0,1) 
plt.title("training error convergence curve with dynamic learning rate")
plt.plot(e)

# get the test error in test set
pred = bpn2.PredLabel(X_test);
count  = 0
for i in range(len(y_test)) :
    if pred[i] == y_test[i]: count += 1

zhengquelv = count/len(y_test)
print("正确率: %.6f" % zhengquelv)

plt.show()

