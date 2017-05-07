
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris=pd.read_csv('./data/iris.csv')
shuffled_rows=np.random.permutation(iris.index)
iris=iris.loc[shuffled_rows,:]
print(iris.shape)

get_ipython().magic('matplotlib inline')
print(iris.species.unique())
iris.hist(["sepal_length","sepal_width","petal_length","petal_width"])
plt.show()



iris['ones']=np.ones(iris.shape[0])
X=iris[['ones', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
Y=((iris['species']=='Iris-versicolor').values.astype(int))

def sigmoid_activation(x, theta):
    x = np.asarray(x)
    theta = np.asarray(theta)
    #return 1 / (1 + np.exp(-np.dot(theta.T, x)))
    return 1 / (1 + np.exp(-np.dot(x, theta)))
#Calculate  cost
def cal_cost(x,y,theta):
    #h=sigmoid_activation(x.T,theta)
    h=sigmoid_activation(x,theta).T
    cost=-np.mean(y*(np.log(h))+(1-y)*np.log(1-h))  #h应该为1×n的样式，这样y×h才能得到一个向量而非矩阵
    return cost #计算的是累计误差，即所有样本的误差

#计算误差的梯度
def gradient(X,Y,theta):
    grads=np.zeros(theta_init.shape)  #5*1
    for i,obs in enumerate(X):
        h=sigmoid_activation(obs,theta)  #计算单个样例的梯度，再把所有样例的累加起来然后取平均值，也可以直接计算所有样例的梯度，再取平均值。
        delta=(Y[i]-h)*h*(1-h)*obs  #为（5,）的一个array
        grads+=delta[:,np.newaxis]/len(X)  #需要对delta增加维度才能与grads相加
    return grads



#接下来我们给出单层神经网络的完整theta计算过程：
theta_init = np.random.normal(0,0.01,size=(5,1))
def learn(X,Y,theta,alpha=0.1):
    
    counter=0
    max_iteration=1000
    
    convergence_thres=0.000001
    c=cal_cost(X,Y,theta)
    cost_pre=c+convergence_thres+0.01
    costs=[c]
    while( (np.abs(c-cost_pre)>convergence_thres) & (counter<max_iteration)):
        grads=gradient(X,Y,theta)
        theta+=alpha*grads
        cost_pre=c
        c=cal_cost(X,Y,theta)
        costs.append(c)
        counter+=1
    return theta,costs

theta,costs=learn(X,Y,theta_init)
plt.plot(costs)
plt.title("Convergence of the Cost Function")
plt.ylabel("J($\Theta$)")
plt.xlabel("Iteration")
plt.show()




