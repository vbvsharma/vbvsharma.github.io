---
layout: post
title: Logistic Regression with a Neural Network Mindset
category: "Neural Networks"
author: Vaibhav Sharma
comments: true
mathjax: true
---

Today we are going to implement logistic regression as a neural network. This is definitely one of the simplest neural network, and is great to get your feet wet in neural network. After completing this tutorial, you will know:
* How to implement logistic regression.
* How to use logistic regression.
* How to use gradient descent.
* How a neural network works.
* Case study - Breast Cancer Wisconsin Data Set (predict whether the tumor is benign or malignant)

## Introduction to Logistic Regression

[Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) is a binary classification algorithm, which can be used to classify linearly seperable classes, i.e. two classes that can be separated by a line. It is one of the most basic algorithms. Most of them time, this is the first algorithm that I use for classification. It kind of provides the lower bound for accuracy. Let us see the architecture of logistic regression.

# The General Architecture of the learning algorithm

Before you get started please check out the **[notation](https://d3c33hcgiwev3.cloudfront.net/_106ac679d8102f2bee614cc67e9e5212_deep-learning-notation.pdf?Expires=1512518400&Signature=DZK13JO29O-sgV8oBeaiFO0eOS5u~DqDAo4xnxdNjnueZJyUQCCDYMOCRTaZOxcZCvgtLZb0Q89ZVHIBDtZ1HbfKGEV8GG-aZ1ban5VnWAlKNqjBnUfxrmHqaavc111CHMz3OIqalaD~qyCy~JlFkuDEVl1Yunlx9SNS2hPrDKA_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)** that is being used below.
The following image explains the architecture of logistic regression as a neural network:


![Logistic Regression as a Neural Network](/assets/images/2019-06-29-Logistic-Regression-with-a-Neural-Network-Mindset/Logistic-Regression-as-a-Neural-Network.png){:class="img-responsive"}


**Mathematical expression of the algorithm**

For one example $$x^{(i)}$$:

$$z^{(i)} = w^T x^{(i)} + b$$

$$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})$$ 

$$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})$$

The cost is then computed by summing over all training examples:
$$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})$$

**Key steps**:
We will carry out the following steps: 
    - Initialize the parameters of the model
    - Learn the parameters for the model by minimizing the cost  
    - Use the learned parameters to make predictions (on the test set)
    - Analyse the results and conclude

## Building the parts of our algorithm

The main steps for building a Neural Network are:
1. Define the model structure (such as number of input features) 
2. Initialize the model's parameters
3. Loop:
    - Calculate current loss (forward propagation)
    - Calculate current gradient (backward propagation)
    - Update parameters (gradient descent)

You often build 1-3 separately and integrate them into one function we call `model()`.


```python
# import the libraries needed
import numpy as np
```

### **Helper functions**
Implement `sigmoid()` function to compute $sigmoid( w^T x + b)$ to make predictions.


```python
def sigmoid(z):
    """
    Computer the sigmoid of z
    
    Arguments:
    x -- A scalar or numpy array of any size.
    
    Return:
    s -- sigmoid(z)
    """
    
    # Calculate sigmoid
    s = 1 / (1 + np.exp(-z))
    
    return s
```


```python
# function to test sigmoid
def test_sigmoid():
    print ("sigmoid(0) = " + str(sigmoid(0)))
    print ("sigmoid(9.2) = " + str(sigmoid(9.2)))

test_sigmoid()
```
**Output**
```
    sigmoid(0) = 0.5
    sigmoid(9.2) = 0.999898970806
```

### **Initializing parameters**
Implement parameter initialization to initialize w as a vector of zeros and b to zero.


```python
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    # Initialize `w` and `b`
    w = np.zeros(shape=(dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b
```


```python
def test_initialize_with_zeros(dim):
    w, b = initialize_with_zeros(dim)
    print ("w = " + str(w))
    print ("b = " + str(b))

test_initialize_with_zeros(2)
```
**Output**
```
    w = [[ 0.]
     [ 0.]]
    b = 0
```

### **Forward and Backward propagation**
Now that your parameters are initialized, you can do the "forward" and "backward" propagation steps for learning the parameters.
Forward Propagation:
- You get X
- You compute $$A = \sigma(w^T X + b) = (a^{(0)}, a^{(1)}, ..., a^{(m-1)}, a^{(m)})$$
- You calculate the cost function: $$J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})$$

Here are the two formulas you will be using: 

$$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T$$

$$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})$$


```python
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array
    b -- bias, a scalar
    X -- data of size (number of features, number of examples)
    Y -- true "label" vector of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """
    
    m = X.shape[1]
    
    ### Forward Propagation
    # compute activation
    A = sigmoid(np.dot(w.T, X) + b)
    # compute cost
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
    
    ### Backward Propagation
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
```


```python
def test_propagate():
    w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
    grads, cost = propagate(w, b, X, Y)
    print ("dw = " + str(grads["dw"]))
    print ("db = " + str(grads["db"]))
    print ("cost = " + str(cost))

test_propagate()
```
**Output**
```
    dw = [[ 0.99993216]
     [ 1.99980262]]
    db = 0.499935230625
    cost = 6.00006477319
```

### **Optimization**
- You have initialized your parameters.
- You are also able to compute a cost function and its gradient.
- Now, you want to update the parameters using gradient descent.

Implementing the optimization function. The goal is to learn $w$ and $b$ by minimizing the cost function $J$. For a parameter $\theta$, the update rule is $ \theta = \theta - \alpha \text{ } d\theta$, where $\alpha$ is the learning rate.


```python
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (number of features, 1)
    b -- bias, a scalar
    X -- data of shape (number of features, number of examples)
    Y -- true "label" vector (containing 0 or 1), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update weights and bias
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
```


```python
def test_optimize():
    w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
    
    params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
    
    print ("w = " + str(params["w"]))
    print ("b = " + str(params["b"]))
    print ("dw = " + str(grads["dw"]))
    print ("db = " + str(grads["db"]))

test_optimize()
```
**Output**
```
    w = [[ 0.1124579 ]
     [ 0.23106775]]
    b = 1.55930492484
    dw = [[ 0.90158428]
     [ 1.76250842]]
    db = 0.430462071679
```

### **Predict**

The previous function will output the learned w and b. We are able to use w and b to predict the labels for a dataset X. Implement the `predict()` function. There is two steps to computing predictions:

1. Calculate $$\hat{Y} = A = \sigma(w^T X + b)$$

2. Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5), stores the predictions in a vector `Y_prediction`.


```python
def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (number of features, 1)
    b -- bias, a scalar
    X -- data of size (number of features, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """
    
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of class being "1"
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        # Convert probabilities a[0,i] to actual predictions p[0,i]
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction
```


```python
def test_predict():
	w, b, X = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]])
    print("predictions = " + str(predict(w, b, X)))

test_predict()
```

**Output**
```
predictions = [[ 1.  1.]]
```

**What to remember:**
You've implemented several functions that:
- Initialize (w,b)
- Optimize the loss iteratively to learn parameters (w,b):
    - computing the cost and its gradient 
    - updating the parameters using gradient descent
- Use the learned (w,b) to predict the labels for a given set of examples

### **Merge all functions into a model**

You will now see how the overall model is structured by putting together all the building blocks (functions implemented in the previous parts) together, in the right order.

Implement the model function. Use the following notation:
    - Y_prediction for your predictions on the test set
    - Y_prediction_train for your predictions on the train set
    - w, costs, grads for the outputs of optimize()


```python
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)


    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
```

We have successfully implemented logistic regression as a neural network. Now, let us use it for prediction.

## Case Study - Breast Cancer Wisconsin Data Set

[Breast Cancer Wisconsin (Diagnostic) Data Set](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)'s features can be used to predict the type of tumor, malignant or benign. You can check the data set's description [here](http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names) and download it from [here](http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data). Its attributes are as follows:

#### **Attribute Information**
  1) ID number
  
  2) Diagnosis (M = malignant, B = benign)
  
  3-32) Other attributes

Now, let us load this data set.


```python
import pandas as pd

data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header=None)
print(data.shape)
```
**Ouput**
```
(569, 32)
```
So we have 569 rows and 32 coulmns.

Let us check what the data looks like?


```python
# Explore a bit
data.head()
```
**Ouput**

<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>



In this set, the 0<sup>th</sup> column is `id`, 1<sup>st</sup> is the class, `B` for benign and `M` for malignant, and further columns are real-valued input features. Let us gather the input features and corresponding output classes from data.


```python
# Extract input features from column 2 to 32
X = data.iloc[:, 2:]
# Extract diagnosis class column 1
Y = data.iloc[:, 1]
```


```python
# Changing M to 1 and B to 0, for malignant and benign respectively
Y = (Y == 'M').astype(np.float64)
```


```python
# Changing DataFrame to numpy arrays
X = X.values
Y = Y.values

# Changing shape of Y 
Y = np.resize(Y, (Y.shape[0], 1))

# Normalize the data features
from sklearn.preprocessing import normalize
X = normalize(X, axis=0)
```

While working with neural networks and optimization algorithms, always normalize your data features. This is done for following two reasons:
* It prevents the values involved in a network from becoming too large or too small, hence reduces the chances of overflow or underflow
* It helps the optimizer to converge (or approach convergence) faster.

To see what happens when you don't normalize your data, try commenting out those lines and run the code. (You will get cost's value nan for nearly all epochs).


```python
# Prepare training and test data
# Put 70% of examples in training set and the remaing 30% in testing set 
X_train = X[:int(0.7*X.shape[0]), :].T
Y_train = Y[:int(0.7*Y.shape[0]), :].T
X_test = X[int(0.7*X.shape[0]):, :].T
Y_test = Y[int(0.7*Y.shape[0]):, :].T
```


```python
# Let's run the model now!!!
model(X_train, Y_train, X_test, Y_test, print_cost=True)
```

**Ouput**
```
    Cost after iteration 0: 0.693147
    Cost after iteration 100: 0.647681
    Cost after iteration 200: 0.614502
    Cost after iteration 300: 0.585481
    Cost after iteration 400: 0.559981
    Cost after iteration 500: 0.537452
    Cost after iteration 600: 0.517429
    Cost after iteration 700: 0.499527
    Cost after iteration 800: 0.483430
    Cost after iteration 900: 0.468875
    Cost after iteration 1000: 0.455647
    Cost after iteration 1100: 0.443567
    Cost after iteration 1200: 0.432484
    Cost after iteration 1300: 0.422276
    Cost after iteration 1400: 0.412835
    Cost after iteration 1500: 0.404075
    Cost after iteration 1600: 0.395917
    Cost after iteration 1700: 0.388299
    Cost after iteration 1800: 0.381163
    Cost after iteration 1900: 0.374461
    train accuracy: 90.20100502512562 %
    test accuracy: 95.32163742690058 %





    {'Y_prediction_test': array([[ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,
              0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,
              0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,
              0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,
              0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,
              0.,  1.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,
              1.,  0.]]),
     'Y_prediction_train': array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,
              0.,  1.,  1.,  0.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,
              1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,
              0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,
              0.,  1.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,
              1.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,
              1.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,
              0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,
              1.,  1.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,
              0.,  1.,  1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,
              0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
              1.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  1.,  1.,
              0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,
              0.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,
              0.,  0.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,
              0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  1.,
              0.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
              1.,  0.,  0.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,
              1.,  0.,  1.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
              0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  0.,
              0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,
              0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,
              0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
              0.,  1.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  0.,  0.,  1.,
              0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
              0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.]]),
     'b': -2.82004015631605,
     'costs': [0.69314718055994518,
      0.64768121608659124,
      0.61450221434667129,
      0.58548108050436609,
      0.55998108981562744,
      0.53745153619366792,
      0.51742861842701759,
      0.49952734366191881,
      0.4834301607574058,
      0.46887545101664851,
      0.45564724081802283,
      0.44356654183651678,
      0.43248428219793783,
      0.42227562488511811,
      0.41283543232151004,
      0.40407465235038642,
      0.395917434789408,
      0.38829882374096664,
      0.3811629029392809,
      0.37446129799735056],
     'learning_rate': 0.5,
     'num_iterations': 2000,
     'w': array([[ 2.2402501 ],
            [ 1.46136967],
            [ 2.39127058],
            [ 4.24369791],
            [ 0.58075704],
            [ 3.17224642],
            [ 5.31606193],
            [ 6.06826111],
            [ 0.47147587],
            [-0.25072496],
            [ 3.70336513],
            [-0.19500842],
            [ 3.66594003],
            [ 4.71289032],
            [-0.68066637],
            [ 1.12038991],
            [ 0.69818476],
            [ 1.72171634],
            [-0.62144059],
            [-0.66633135],
            [ 2.92714447],
            [ 1.82405968],
            [ 3.06332473],
            [ 5.21202012],
            [ 1.06115042],
            [ 4.13504157],
            [ 4.99853943],
            [ 5.33078157],
            [ 1.25268158],
            [ 0.8297114 ]])}
```


We get,

train accuracy: 90.20 %

test accuracy: 95.32 %

Not bad for such a simple algorithm.

Well done for completing the tutorial. Hope you liked it.

The following is the complete code of the application of the above model.


```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header=None)

# Extract input features from column 2 to 32
X = data.iloc[:, 2:]
# Extract diagnosis class column 1
Y = data.iloc[:, 1]

# Changing M to 1 and B to 0, for malignant and benign respectively
Y = (Y == 'M').astype(np.float64)

# Changing DataFrame to numpy arrays
X = X.values
Y = Y.values

# Changing shape of Y 
Y = np.resize(Y, (Y.shape[0], 1))

# Normalize the data features
X = normalize(X, axis=0)

# More preprocessing, preaparing data to be fed to our model
X_train = X[:int(0.7*X.shape[0]), :].T
Y_train = Y[:int(0.7*Y.shape[0]), :].T
X_test = X[int(0.7*X.shape[0]):, :].T
Y_test = Y[int(0.7*Y.shape[0]):, :].T

# Let's run the model now!!!
model(X_train, Y_train, X_test, Y_test, print_cost=True)
```

Things left for another day:
* Analysis of logistic regression.
* How to choose appropriate threshold for the logistic regression.
* Prevent overfitting and underfitting (Regularization, one of the methods).
* Why logistic regression performs poorly on data whose classes cannot be linearly separated.
* How to tweak logistic regression to support multiclass classification.
* After getting 95% test accuracy, why we need to experiment more to get accurate results (cross-validation).
