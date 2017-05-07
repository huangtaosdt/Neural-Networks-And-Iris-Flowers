# Import dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris=pd.read_csv('./data/iris.csv')
shuffled_rows=np.random.permutation(iris.index)
iris=iris.loc[shuffled_rows,:]
# print(iris.shape)

# %matplotlib inline
# print(iris.species.unique())
# iris.hist(["sepal_length","sepal_width","petal_length","petal_width"])
# plt.show()

iris['ones']=np.ones(iris.shape[0])
X=iris[['ones', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y=((iris['species']=='Iris-versicolor').values.astype(int))

class NNet:
    def __init__(self,learning_rate=0.5,maxepochs=1e4,convergence_thres=1e-5,hidden_layer=4):
        self.learning_rate=learning_rate
        self.maxepochs=int(maxepochs)
        self.convergence_thres=convergence_thres
        self.hidden_layer=int(hidden_layer)
        
    def _sigmoid_activation(self,X,theta):
        x = np.asarray(X)
        theta = np.asarray(theta)
        return 1 / (1 + np.exp(-np.dot(x, theta)))
    
    def _multi_cost(self,X,y):
        # feed through network
        l1,l2=self._feedforward(X)
        # compute error
        inner=y*np.log(l2)+(1-y)*np.log(1-l2)
        # negative of average error
        return -np.mean(inner)
    
    def _feedforward(self,X):
        l1 = sigmoid_activation(X, self.theta0)  #100*4
        l1 = np.column_stack([np.ones(l1.shape[0]), l1]) #100*5
        l2 = sigmoid_activation(l1, self.theta1).T  #1*100
        return l1,l2
    def predict(self,X):
        _,y=self._feedforward(X)
        return y
    
    def learn(self,X,y):
        counter=0
        nobs,ncols=X.shape
        self.theta0=np.random.normal(0,0.01,size=(ncols,self.hidden_layer))
        self.theta1=np.random.normal(0,0.01,size=(self.hidden_layer+1,1))
        c=self._multi_cost(X,y)
        cost_pre=c+self.convergence_thres+0.01
        self.costs=[c]
        
        while( (np.abs(c-cost_pre)>self.convergence_thres) & (counter<self.maxepochs)):
            
            l1,l2=self._feedforward(X)  # l1-100*5,  l2:1*100 的数组
            
            l2_delta=(y-l2)*l2*(1-l2)  #delta:权值更新公式中的一部分
            l1_delta=l2_delta.T.dot(self.theta1.T)*l1*(1-l1) #dot(l2_delta.T,self.theta1.T)，100*1×1*n,得到一个100*5的矩阵，100个样例，隐藏层由5个神经元
            self.theta1+=np.dot(l1.T,l2_delta.T)/nobs*self.learning_rate
            self.theta0+=np.dot(l1_delta.T,X)[:,1:]/nobs*self.learning_rate  #np.dot(l1_delta.T,X) 得5*5矩阵；[:,1:]去掉每行的第一列
            cost_pre=c
            c=self._multi_cost(X,y)
            self.costs.append(c)
            counter+=1

#l1 shape: (100, 5)
# l2 shape: (1, 100)
# l1_delta shape: (100, 5)
# l2_delta shape: (1, 100)

learning_rate = 0.5
maxepochs = 10000       
convergence_thres = 0.00001  
hidden_units = 4

# Initialize model
model = NNet(learning_rate=learning_rate, maxepochs=maxepochs,
              convergence_thres=convergence_thres, hidden_layer=hidden_units)
# Train model -- get optimal thetas.
model.learn(X, y)
prediction=model.predict(X)
prediction=np.array([i>=0.5 for i in prediction]).astype(int)

plt.plot(model.costs)
plt.title("Convergence of the Cost Function")
plt.ylabel("J($\Theta$)")
plt.xlabel("Iteration")
plt.show()