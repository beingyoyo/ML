import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np

def computeCost(X,y,thetha): #Cost func
    temp=np.dot(X,thetha)-y
    return np.sum(np.power(temp,2))/2*m

def gradientDescent(X,y,thetha,alpha,iterations): #gradient desc for optimal params
    for _ in range(iterations):
        temp=np.dot(X,thetha)-y
        temp=np.dot(X.T,temp)
        thetha=thetha-(alpha/m)*temp
    return thetha

data=pd.read_csv("data2.csv")#Reading file
#printing desc
print(data.columns)
print(data.shape)

X=data.iloc[:,0] #first column
y=data.iloc[:,1]  #second column
m=len(y) #length of data

#plotting data
plt.scatter(X,y,color="b",marker="*")
plt.xlabel("population in 10,000's")
plt.ylabel("Profit in $10,000's")
plt.show()

X=X[:,np.newaxis]
y=y[:,np.newaxis]
thetha=np.zeros([2,1])
iterations=1500
alpha=0.01
ones=np.ones((m,1))
X=np.hstack((ones,X))

J=computeCost(X,y,thetha) #Computing cost
print(J)

thetha=gradientDescent(X,y,thetha,alpha,iterations) #Finding optimal params
print(thetha)

J=computeCost(X,y,thetha) #Computing cost
print(int(J))

plt.scatter(X[:,1],y,color="b",marker="*")
plt.xlabel("population in 10,000's")
plt.ylabel("Profit in $10,000's")
plt.plot(X[:,1],np.dot(X,thetha))
plt.show()
