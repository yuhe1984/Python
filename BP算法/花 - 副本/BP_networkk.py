'''
the definition of BP network class
'''
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cholesky
import math
import copy
random.seed(10)
def rand(a,b):
    return (b-a)*random.random()+a
#random.random是生成一个(0,1)之间的浮点数

class BP_networkk: 

    def __init__(self):
        
        '''
        initial variables
        '''
        # node number each layer
        self.i_n = 0           
        self.h_n = 0   
        self.o_n = 0
        self.ee = float('inf')
        self.ek = float('inf')
        # output value for each layer
        self.i_v = []       
        self.h_v = []
        self.o_v = []

        # parameters (w, t)
        self.ih_w = []    # weight for each link
        self.ho_w = []
        self.h_t  = []    # threshold for each neuron
        self.o_t  = []

        # definition of alternative activation functions and it's derivation
        self.fun = {
            'Sigmoid': Sigmoid, 
            'SigmoidDerivate': SigmoidDerivate,
            'Tanh': Tanh, 
            'TanhDerivate': TanhDerivate,
            
            # for more, add here
            }
        
        # initial the learning rate
        self.lr1 = []  # output layer
        self.lr2 = []  # hidden layer
        self.e1 = []
        self.i = 0
        self.b1 = []
        self.d1 = []

        # output value for each layer
        self.i_vv = []       
        self.h_vv = []
        self.o_vv = []

        # parameters (w, t)
        self.w1 = []    # weight for each link
        self.w2 = []
        self.t1 = []    # threshold for each neuron
        self.t2 = []

        # parameters (w, t)
        self.ww1 = []
        self.ww2 = []
        self.tt1 = []
        self.tt2 = []
        self.y1 = []
        self.i1 = 0
        
        
    def CreateNN(self, ni, nh, no, actfun, learningrate):
        '''
        build a BP network structure and initial parameters
        @param ni, nh, no: the neuron number of each layer
        @param actfun: string, the name of activation function
        @param learningrate: learning rate of gradient algorithm
        '''
        
        # dependent packages
        import numpy as np       
               
        # assignment of node number
        self.i_n = ni
        self.h_n = nh
        self.o_n = no
        
        # initial value of output for each layer
        self.i_v = np.zeros(self.i_n)
        self.h_v = np.zeros(self.h_n)
        self.o_v = np.zeros(self.o_n)

        # initial weights for each link (random initialization)
        self.ih_w = np.zeros([self.i_n, self.h_n])
        self.ho_w = np.zeros([self.h_n, self.o_n])
        for i in range(self.i_n):  
            for h in range(self.h_n):
                self.ih_w[i][h] = rand(0,1)
                #print(self.ih_w[i][h])
                
        for h in range(self.h_n):  
            for j in range(self.o_n):
                self.ho_w[h][j] = rand(0,1)
                #print(self.ho_w[h][j])
                
        # initial threshold for each neuron
        self.h_t = np.zeros(self.h_n)
        self.o_t = np.zeros(self.o_n)
        for h in range(self.h_n):
            self.h_t[h] = rand(0,1)
            #print(self.h_t[h])
        for j in range(self.o_n):
            self.o_t[j] = rand(0,1)
            #print(self.o_t[j])

        # initial activation function
        self.af  = self.fun[actfun]
        self.afd = self.fun[actfun+'Derivate']

        # initial learning rate
        self.lr=learningrate
        

    def Pred(self, x):
        '''
        predict process through the network
        @param x: the input array for input layer
        '''
        
        # activate input layer
        for i in range(self.i_n):
            self.i_v[i] = x[i]
            
        # activate hidden layer
        for h in range(self.h_n):
            total = 0.0
            for i in range(self.i_n):
                total += self.i_v[i] * self.ih_w[i][h]
            self.h_v[h] = self.af(total - self.h_t[h])
            
        # activate output layer
        for j in range(self.o_n):
            total = 0.0
            for h in range(self.h_n):
                total += self.h_v[h] * self.ho_w[h][j]
            self.o_v[j] = self.af(total - self.o_t[j])

        #print(self.o_v)

 
    '''
    for fixed learning rate
    '''    
        
    def BackPropagate(self, x, y):
        '''
        the implementation of BP algorithms on one slide of sample
        
        @param x, y: array, input and output of the data sample
        '''
        
        # dependent packages
        import numpy as np 

        # get current network output
        self.Pred(x)
        
        # calculate the gradient based on output
        o_grid = np.zeros(self.o_n) 
        for j in range(self.o_n):
            o_grid[j] = (y[j] - self.o_v[j]) * self.afd(self.o_v[j])
        
        h_grid = np.zeros(self.h_n)
        for h in range(self.h_n):
            for j in range(self.o_n):
                h_grid[h] += self.ho_w[h][j] * o_grid[j]
            h_grid[h] = h_grid[h] * self.afd(self.h_v[h])   

        # updating the parameter
        for h in range(self.h_n):  
            for j in range(self.o_n): 
                self.ho_w[h][j] += self.lr * o_grid[j] * self.h_v[h]
           
        for i in range(self.i_n):  
            for h in range(self.h_n): 
                self.ih_w[i][h] += self.lr * h_grid[h] * self.i_v[i]     

        for j in range(self.o_n):
            self.o_t[j] -= self.lr * o_grid[j]    
                
        for h in range(self.h_n):
            self.h_t[h] -= self.lr * h_grid[h]

    '''def error(self,e_k):
        
        import numpy as np
        self.ih_w = self.ih_w.tolist()
        self.ho_w = self.ho_w.tolist()
        self.h_t = self.h_t.tolist()
        self.o_t = self.o_t.tolist()
        self.ww1.append(self.ih_w)
        self.ww2.append(self.ho_w)
        self.tt1.append(self.h_t)
        self.tt2.append(self.o_t)
        self.ih_w = np.array(self.ih_w)
        self.ho_w = np.array(self.ho_w)
        self.h_t = np.array(self.h_t)
        self.o_t = np.array(self.o_t)
        self.lr2.append(self.lr)
        
        err=sum(e_k)/len(e_k)

        self.e1.append(err)
        
        if rand(0,1) < 0.01:
            self.lr = rand(0,1)
            #print(self.lr)
        #elif err-self.ee < 0 :#ee为上一步误差
        else: 
            if err - self.ee < 0:
               self.lr1.append(self.lr)
               #print(self.lr1)
            #if len(self.lr1) > 0:
               a=np.array(self.lr1)
               b=np.mean(a)
               c=np.var(a)
               d=math.sqrt(c)
               self.b1.append(b)
               self.d1.append(d)
               self.lr=random.normalvariate(b,d)
               self.y1.append(len(self.e1)-1)
               #self.lr=random.gauss(b,c)
               self.lr1=a.tolist()
               #print(self.lr)
            if err - self.ee >= 0:
               self.ih_w = np.array(self.ww1[self.e1.index(min(self.e1))-1])
               self.ho_w = np.array(self.ww2[self.e1.index(min(self.e1))-1])
               self.h_t = np.array(self.tt1[self.e1.index(min(self.e1))-1])
               self.o_t = np.array(self.tt2[self.e1.index(min(self.e1))-1])
               self.lr = self.lr2[self.e1.index(min(self.e1))-1]
               #print(self.lr)
        self.ee=err
        #print(err)
        #print(self.lr)
        return err'''
   
    def TrainStandard(self, data_in, data_out):
        '''
        standard BP training
        @param lr, learning rate, default 0.05
        @return: e, accumulated error
        @return: e_k, error array of each step
        '''    
        e_k = []
        for k in range(len(data_in)):
            x = data_in[k]
            y = data_out[k]
            self.BackPropagate(x, y)
            # error in train set for each step

            y_delta2 = 0.0
            for j in range(self.o_n):
                y_delta2 += (self.o_v[j] - y[j]) * (self.o_v[j] - y[j])
            e_k.append(y_delta2/2)
            if self.i1 == 0:
                self.y1.append(y_delta2/2)
        if self.i1 > 0:
            for i in range(len(e_k)):
                if self.y1[i] > e_k[i]:
                    self.y1[i] = e_k[i]
                    self.lr1.append(self.lr)
            if rand(0,1) < 0.01:
                self.lr = rand(0,1)
                #print('self.lr',self.lr)
                #self.lr1.append(self.lr)
            else:
                a=np.array(self.lr1)
                b=np.mean(a)
                c=np.var(a)
                d=math.sqrt(c)
                self.lr=random.normalvariate(b,d)
                #self.lr=random.gauss(b,c)
                self.lr1=a.tolist()
                #print(self.lr)
        self.i1 += 1
        # total error of training
        e = sum(e_k)/len(e_k)
        #e=self.error(e_k)
        #print(self.lr)
        return e, e_k
    
    def Prede(self, x):        
        # activate input layer
        for i in range(self.i_n):
            self.i_v[i] = x[i]
            
        # activate hidden layer
        for h in range(self.h_n):
            total = 0.0
            for i in range(self.i_n):
                total += self.i_v[i] * self.ih_w[i][h]
            self.h_v[h] = self.af(total - self.h_t[h])
            
        # activate output layer
        for j in range(self.o_n):
            total = 0.0
            for h in range(self.h_n):
                total += self.h_v[h] * self.ho_w[h][j]
            self.o_v[j] = self.af(total - self.o_t[j])

        print(self.o_v)

    def mean(self):
        return self.b1

    '''def svar(self):
        return self.d1

    def time(self):
        return self.y1
   
    def printf(self, data_in,):
        for k in range(len(data_in)):
            x = data_in[k]
            self.Prede(x)'''
            
    def PredLabel(self, X):
        '''
        predict process through the network
        
        @param X: the input sample set for input layer
        @return: y, array, output set (0,1,2... - class) based on [winner-takes-all] 
        '''    
        import numpy as np
               
        y = []
        
        for m in range(len(X)):
            self.Pred(X[m])
#             if self.o_v[0] > 0.5:  y.append(1)
#             else : y.append(0)
            max_y = self.o_v[0]
            label = 0
            for j in range(1,self.o_n):
                if max_y < self.o_v[j]:
                    label = j
                    max_y = self.o_v[j]
            y.append(label)
           
        return np.array(y)
    
'''
the definition of activation functions
'''
def Sigmoid(x):
    '''
    definition of sigmoid function and it's derivation
    '''
    from math import exp
    return 1.0 / (1.0 + exp(-x))
def SigmoidDerivate(y):
    return y * (1 - y)

def Tanh(x):
    '''
    definition of sigmoid function and it's derivation
    '''
    from math import tanh
    return tanh(x)
def TanhDerivate(y):
    return 1 - y*y
