import matplotlib.pyplot as plt
import numpy as np # 导入numpy并命名为np
x = np.mat( '0,0,1,1;\
             0,1,0,1\
             ')
x = np.array(x).T
y=np.mat('0,1,1,0')
y = np.array(y).T
xrow, xcol = x.shape
yrow, ycol = y.shape
#experiment of fixed learning rate
from BP_network import *
bpn1 = BP_network()  
bpn1.CreateNN(xcol, 5, ycol, actfun = 'Sigmoid', learningrate = 1)
e = []
for i in range(1000): 
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
bpn2.CreateNN(xcol, 5, ycol, actfun = 'Sigmoid', learningrate = 1)
e = []
for i in range(1000): 
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


