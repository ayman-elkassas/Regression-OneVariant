# libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Regression without any ML lib (pure math)

# read data
path='week_1-ex_1.txt'
# there is no header in csv file (csv not important as .csv ext but is comma separator)
data=pd.read_csv(path,header=None,names=['Population','Profit'])

# show data details
# head as pointer of file stop on line 10 from start
print('data=\n',data.head(10))
print("****************************************************")
print('data.describe=\n',data.describe())
# describe is a function that get mean, std, min, max and etc for each feature.
print("****************************************************")

# draw data
data.plot(kind='scatter',x='Population',y='Profit',figsize=(5,5))
plt.show()

# after show data, we can say that decision boundary may be as linear equation
# and your problem in one feature as population as x and profit output as y

# then (linear equation in one variant y= theta0 + theta1 * x)

data.insert(0,'Ones',1)
print('data=\n',data.head(10))
print("****************************************************")

# separate x (training data) from y as target value
# shape => to get 97*3 dataset
cols=data.shape[1]
X=data.iloc[:,0:cols - 1]
Y=data.iloc[:,cols - 1:cols]

print('X data=\n',X.head(10))
print('Y data=\n',Y.head(10))
print("****************************************************")

# convert to matrix
x=np.matrix(X.values)
y=np.matrix(Y.values)
theta=np.matrix(np.array([0,0]))


# print('X \n',X)
# print('X.shape = ' , X.shape)
# print('theta \n',theta)
# print('theta.shape = ' , theta.shape)
# print('y \n',y)
# print('y.shape = ' , y.shape)
# print('**************************************')

# cost error function on theta=[0,0]
def computeCost(X,y,theta):
    z=np.power(((x * theta.T) - y),2)

    return np.sum(z) / (2 * len(X))


# now we can compute theta vector from  gradient descent and then compute J
# after nums of iterations

def gradientDescent(x, y, theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    # nums of theta as len of vector
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error=(x * theta.T) - y

        for j in range(parameters):
            term=np.multiply(error,x[:,j])
            temp[0,j]=theta[0,j] - ((alpha / len(x)) * np.sum(term))

        theta = temp
        cost[i]=computeCost(x,y,theta)

    return theta,cost

# initialize variables for learning rate and iterations
alpha = 0.01
iters = 1000

# perform gradient descent to "fit" the model parameters
g, cost = gradientDescent(x, y, theta, alpha, iters)

print('g = ' , g)
print('cost  = ' , cost[0:50] )
print('computeCost = ' , computeCost(x, y, g))
print('**************************************')

# get best fit line

x = np.linspace(data.Population.min(), data.Population.max(), 100)
print('x \n',x)
print('g \n',g)

f = g[0, 0] + (g[0, 1] * x)
print('f \n',f)

# draw the line

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')

ax.scatter(data.Population, data.Profit, label='Training Data')

ax.legend(loc=2)

ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

plt.show()

# draw error graph

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

plt.show()
