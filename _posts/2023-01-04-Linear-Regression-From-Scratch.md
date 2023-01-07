---
layout: post
title: "Linear Regression From Scratch"
category: misc
comments: true
mathjax: true
---

Linear regression is one of the most basic algorithms in machine learning and statistics, and it is also one of the best understood algorithms out there. Here, we are going to study multivariate linear regression, which is just just a fancy name for linear regression when multiple independent variables are involved. We do this for two reasons:
* Multivariate Linear Regression is more general than Univariate Linear Regression.
* Practically you will find yourself using Multivariate Linear Regression more often than Univariate Linear Regression, as it involves learning from multiple features. 

Here you will discover:
* What is linear regression?
* How is it implemented?
* How to estimate linear regression coefficients using gradient descent?
* Case study - Inferring Price of House

Let us get started.

## Introduction to Linear Regression

Linear regression is defined as a linear relationship between a dependent variable (say $$ y $$) and one or more independent variable (say $$ x_{1},x_{2}, ..., x_{n} $$). This can be written mathematically as folows:

$$y = \theta_{0} + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_{n} x_{n} $$

In the above equation, the value of $$ y $$ is being predicted, $$ x_{1},x_{2}, ..., x_{n} $$ are input features that are being used to predict $$ y $$, and $$ \theta_{0}, \theta_{1}, ..., \theta_{n} $$ are called linear regression coefficients or linear regression parameters. These coefficients can be found using gradient descent. (I call them parameters throughout this post.)

The above linear regression model can also be written in vectorized form:

$$
\boldsymbol{y} = 
\left(\begin{array}{cc}
\theta_{0}\\
\theta_{1}\\
.\\
\theta_{n-1}\\
\theta_{n}\end{array}\right)
\boldsymbol{.}
\left(\begin{array}{cc}
1\\
x_1\\
.\\
x_{n-1}\\
x_{n}\end{array}\right)
$$

## Gradient Descent

Gradient descent is an optimizing algorithm. It is used to choose the parameters which minimizes error on the dataset.

We start by initializing the parameters with random weights and perform gradient descent for some iterations. In each iteration we update the parameters such that it gives lesser error in each iteration. The size of each step during descent is defined by the learning rate, which is denoted by $$ \alpha $$. The final parameters are hopefully best fit estimate. If they are not, we may have to tune the algorithm. But that's a different story.

Let us now formalize our gradient descent algorithm. Suppose we have **m** training examples, each training example has **n** features, and the $$ i^{th} $$ training example is denoted by $$ (x^{(i)}, y^{(i)}) $$. We have,

**Hypothesis:** $$ h_\theta = \theta_{0} + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_{n} x_{n} $$

**Parameters:** $$ \theta_{0}, \theta_{1}, ..., \theta_{n} $$

**Cost function:** $$ J(\theta_{0}, \theta_{1}, ..., \theta_{n}) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}-y^{(i)})^2 $$

Our goal is to reduce the cost of the model at each iteration using gradient descent. The gradient descent algorithm is stated below:

**Gradient Descent:**

Repeat {

$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j}J(\theta_{0}, \theta_{1}, ..., \theta_{n})
$$ 

} (simultaneously update for every $$ j = 0, ..., n $$)

Once we understand the above equations, we can vectorize them.

**Vectorized Hypothesis:** $$ h_\theta = x^T\theta $$ , here x is n+1 dimensional vector. We get this by adding "1" at the first position of feature vector.

**Vectorized Parameters:** $$ \theta $$, here \theta is n+1 dimensional vector.

**Vectorized Cost function:** $$ J(\theta) = \frac{1}{2m} (X\theta - y)^T(X\theta - y) $$, here $$ X $$ is a matrix with each row as an input feature. Therefore, its dimension is $$ m \times (n+1) $$.

**Vectorized Gradient Descent:**

Repeat {

$$
\theta := \theta - \alpha \frac{\partial}{\partial\theta}J(\theta) 
$$

}

Here, 

$$
\frac{\partial}{\partial\theta}J(\theta) = 2X^T(X\theta - y)
$$

Hence, finally we get

Repeat {

$$
\theta := \theta - \frac{\alpha}{m} X^T(X\theta - y)
$$

}

We can write this in Python code as follows:

```python
for it in range(num_iters):
	hypothesis = np.dot(X, theta)
	loss = hypothesis - y
theta = theta - (alpha / m) * np.dot(X.T, loss)
```

## Normal Equations for Linear Regression
There is also a closed-form solution to linear regression. But this method should only be used for small datasets, as it gets very expensive for large datasets. We can find the parameters as:

$$
\theta = (X^TX)^{-1}X^Ty
$$

Using this formula does not require any feature scaling, and you will get an exact solution in one calculation: there is no "loop until convergence" like in radient descent.

## Case Study - Inferring Price of House

Enough of the theory, let us get hands on now! We will be inferring price of houses, by using number of bedrooms and size of house as features.

You can download the code and data from [here](https://github.com/vbvsharma/Linear-Regression-From-Scratch).

### Load Data

We can load data using `genfromtxt` from numpy. We also have to extract features (X) and actual prices (y) from the read data.

```python
import numpy as np

print('Loading data ...\n')

# Load data
data = np.genfromtxt('data.txt', delimiter=',')
m = data.shape[0]
n = data.shape[1]-1
X = data[:, 0:-1].reshape((m, n))
y = data[:, -1].reshape((m, 1))
```

### Pre-process Data
We have to normalize the data so that the gradients don't explode. We only normalize the features (X) and not the actual prices (y).

```python
def featureNormalize(X):
	mu = np.mean(X, axis=0)
	sigma = np.std(X, axis=0)
	X_norm = (X - mu) / sigma

	return X_norm, mu, sigma
```

We also have to add a column of ones to X, so that the intercept is not zero.

```python
# Add intercept term to X
ones_col = np.ones((m, 1))
X = np.hstack((ones_col, X))
```

### Gradient Descent
Now we will perform gradient descent for some iterations, as discussed in the introductory theory.

```python
def gradientDescent(X, y, theta=None, alpha=0.01, num_iters=100):
	m = X.shape[0]
	n = X.shape[1]
	if theta is None:
		theta = np.zeros((n, 1))

	J_history = np.zeros((num_iters, 1))

	for it in range(num_iters):
		hypothesis = np.dot(X, theta)
		loss = hypothesis - y
		theta = theta - (alpha / m) * np.dot(X.T, loss)
		J_history[it] = computeCost(X, y, theta)

	return theta, J_history
```

### Plotting the Convergence Graph
Let us see how the cost of the model changes as we iterate.

```python
from matplotlib import pyplot as plt

# Choose some alpha value
alpha = 0.01
num_iters = 100

# Init theta and run gradient descent
theta = np.zeros((n+1, 1))
theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.plot(range(len(J_history)), J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.savefig('Cost at each iteration.png')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent:')
print(theta)
print()
```

We get the graph below.

![Convergence graph](/assets/images/2019-06-01-Linear-Regression-From-Scratch/Cost-at-each-iteration.png){:class="img-responsive"}

### Estimating price of a house
Let us now use our mdel to predict the price of a 1000 sq-ft, 2 bedroom house.

```python
# Estimate the price of  a 1000 sq-ft, 3 br house
x = np.array([1000, 3])
x_norm = (x - sigma) / mu
x_norm = np.hstack((1, x_norm)).reshape((1, n+1))
price = np.dot(x_norm, theta)

print('Predicted price of a 1000 sq-ft, 3 br house (using gradient descent):', price, '\n')
```

### Using Normal Equations to Find Parameters
We can find the parameters analytically for small datasets. The code follows:

```python
# Load data
data = np.genfromtxt('data.txt', delimiter=',')
m = data.shape[0]
X = data[:, 0:-1]
y = data[:, -1].reshape((m, 1))

# Add intercept term to X
ones_col = np.ones((m, 1))
X = np.hstack((ones_col, X))

theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from normal equations:')
print(theta)
print()

# Estimate the price of a 1650 sq-ft, 3 br house
x = np.array([1, 1650, 3])
price = np.dot(x, theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):', price, '\n')
```

## Complete Code

Here is the complete code:

```python
import numpy as np
from matplotlib import pyplot as plt

def featureNormalize(X):
	"""
	Calculates and returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.

	Args:
		X: It a ndarray which contains features. Each of its row is a training example and
		   each column has an attribute of training examples.
    
    Returns:
    	X: The normalized version of X where
           the mean value of each feature is 0 and the standard deviation
           is 1.
        mu: Contains mean of every column in X.
        sigma: Contains standard deviation of every column in X

	"""
	mu = np.mean(X, axis=0)
	sigma = np.std(X, axis=0)
	X_norm = (X - mu) / sigma

	return X_norm, mu, sigma

def computeCost(X, y, theta):
	"""
	Computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
	
	Args:
		X: Input feature ndarray.
		y: Output array
		theta: Current parameters for linear regression.
	
	Returns:
		J: Computed cost of using theta as parameters for linear regression
		to fit the data points in X and y.
	"""
	hypothesis = np.dot(X, theta)
	loss = hypothesis - y
	J = np.sum(loss ** 2) / (2 * m)
	return J

def gradientDescent(X, y, theta=None, alpha=0.01, num_iters=100):
	"""
	Performs gradient descent to learn theta.

	Args:
		X: Input feature ndarray.
		y: Output array
		theta: Initial parameters for linear regression.
		alpha: The learning rate.
		num_iters: Number of iterations of gradient descent to be performed.

	Returns:
		theta: Updated parameters for linear regression.
		J_history: An array that contains costs for every iteration.
	"""
	m = X.shape[0]
	n = X.shape[1]
	if theta is None:
		theta = np.zeros((n, 1))

	J_history = np.zeros((num_iters, 1))

	for it in range(num_iters):
		hypothesis = np.dot(X, theta)
		loss = hypothesis - y
		theta = theta - (alpha / m) * np.dot(X.T, loss)
		J_history[it] = computeCost(X, y, theta)

	return theta, J_history

def normalEqn(X, y):
	"""
	Computes the closed-form solution to linear regression using
	normal equations.

	Args:
		X: Input feature ndarray.
		y: Output array

	Returns:
		theta: Parameters for linear regression calculated using normal 
		       equations.
	"""
	theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
	return theta

print('Loading data ...\n')

# Load data
data = np.genfromtxt('data.txt', delimiter=',')
m = data.shape[0]
n = data.shape[1]-1
X = data[:, 0:-1].reshape((m, n))
y = data[:, -1].reshape((m, 1))

# Print out some data points
print('First 10 examples from the dataset:')
print('x = ', X[0:10, :], "\ny = ", y[0:10])

print('\nProgram paused. Press enter to continue.')

input()

# Scale features
print('\nNormalizing Features ...')

X, mu, sigma = featureNormalize(X)

# Add intercept term to X
ones_col = np.ones((m, 1))
X = np.hstack((ones_col, X))

# Running gradient descent
print('\nRunning gradient descent ...')

# Choose some alpha value
alpha = 0.01
num_iters = 100

# Init theta and run gradient descent
theta = np.zeros((n+1, 1))
theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.plot(range(len(J_history)), J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.savefig('Cost at each iteration.png')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent:')
print(theta)
print()

# Estimate the price of  a 1650 sq-ft, 3 br house
x = np.array([1650, 3])
x_norm = (x - sigma) / mu
x_norm = np.hstack((1, x_norm)).reshape((1, n+1))
price = np.dot(x_norm, theta)

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):', price, '\n')

print('\nProgram paused. Press enter to continue.')

input()

print('\nSolving with normal equations ...\n')

# Load data
data = np.genfromtxt('data.txt', delimiter=',')
m = data.shape[0]
X = data[:, 0:-1]
y = data[:, -1].reshape((m, 1))

# Add intercept term to X
ones_col = np.ones((m, 1))
X = np.hstack((ones_col, X))

theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from normal equations:')
print(theta)
print()

# Estimate the price of a 1650 sq-ft, 3 br house
x = np.array([1, 1650, 3])
price = np.dot(x, theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):', price, '\n')
```

## References

* Machine Learning course taught by Andrew Ng on Coursera. 
