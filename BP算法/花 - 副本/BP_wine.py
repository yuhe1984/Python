import matplotlib.pyplot as plt
import numpy as np # 导入numpy并命名为np
import pandas as pd
wine = pd.read_csv('wine.csv',encoding='gbk')
X = wine.iloc[:,:13].get_values()
wine.iloc[:,-1] = wine.iloc[:,-1].astype('category')
label = wine.iloc[:,13].values.categories
y = wine.iloc[:,13].get_values()
Y = pd.get_dummies(wine.iloc[:,13]).get_values()
xrow,xcol = X.shape
yrow,ycol = Y.shape
x = X
y = Y

#experiment of fixed learning rate
from BP_network import *
bpn1 = BP_network()  
bpn1.CreateNN(xcol, 10, ycol, actfun = 'Sigmoid', learningrate = 0.1)
e = []
for i in range(5000): 
    err, err_k = bpn1.TrainStandard(x, y)
    e.append(err)
#bpn1.printf(x)
print('固定学习率',err)
 
f1 = plt.figure(1) 
plt.xlabel("epochs")
plt.ylabel("error")
plt.ylim(0,1) 
plt.title("training error convergence curve with fixed learning rate")
plt.plot(e)

#plt.show()

#experiment of dynamic learning rate
from BP_networkk import *
bpn2 = BP_networkk()  
bpn2.CreateNN(xcol, 10, ycol, actfun = 'Sigmoid', learningrate = 0.1)
e = []
for i in range(5000): 
    err, err_k = bpn2.TrainStandard(x, y)
    e.append(err)
#bpn2.printf(x)
print('动态学习率',err)
 
f2 = plt.figure(2) 
plt.xlabel("epochs")
plt.ylabel("error")
plt.ylim(0,1) 
plt.title("training error convergence curve with dynamic learning rate")
plt.plot(e)

plt.show()

