import matplotlib.pyplot as plt
import numpy as np # 导入numpy并命名为np
x = np.mat( '2,3,3,2,1,2,3,3,3,2,1,1,2,1,3,1,2;\
            1,1,1,1,1,2,2,2,2,3,3,1,2,2,2,1,1;\
            2,3,2,3,2,2,2,2,3,1,1,2,2,3,2,2,3;\
            3,3,3,3,3,3,2,3,2,3,1,1,2,2,3,1,2;\
            1,1,1,1,1,2,2,2,2,3,3,3,1,1,2,3,2;\
            1,1,1,1,1,2,2,1,1,2,1,2,1,1,2,1,1;\
            0.697,0.774,0.634,0.668,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719;\
            0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103\
            ')
x = np.array(x).T
y = np.mat('1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0')
y = np.array(y).T
xrow, xcol = x.shape
yrow, ycol = y.shape
#experiment of fixed learning rate
from BP_network import *
bpn1 = BP_network()  
bpn1.CreateNN(xcol, 5, ycol, actfun = 'Sigmoid', learningrate = 0.1)
e = []
for i in range(3000): 
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
bpn2.CreateNN(xcol, 5, ycol, actfun = 'Sigmoid', learningrate = 0.1)
e = []
for i in range(3000): 
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


