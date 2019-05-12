---
layout: post
title: "Build a CNN on CIFAR-10 using TensorFlow"
categories: misc
---


## Introduction

**Note:** You can find the code for this post [here](https://github.com/vbvsharma/build-cnn-on-cifar10-using-tensorflow).

[The CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html) is a standard dataset used in computer vision and deep learning community. It consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The mapping of all 0-9 integers to class labels is listed below.:

- 0 ~> Airplane
- 1 ~> Automobile
- 2 ~> Bird
- 3 ~> Cat
- 4 ~> Deer
- 5 ~> Dog
- 6 ~> Frog
- 7 ~> Horse
- 8 ~> Ship
- 9 ~> Truck

It is a fairly simple dataset. Hence, it provides the flexibility to play with various techniques, suh as hyperparameter tuning, regularization, training-test split, parameter search, etc. Therefore, I encourage the reader to play with this dataset after reading this tutorial.

In this tutorial, we will build a convolutional neural network model from scratch using TensorFlow, train that model and then evaluate its performance on unseen data.

## Explore CIFAR-10 dataset

Let us load the dataset. The dataset is split into training and testing sets. The training set consists of 50000 images, with 5000 images of each class, and the testing set consists of 10000 images, with 1000 images from each class.


```python
# Import the CIFAR-10 dataset from keras' datasets
from tensorflow.keras.datasets import cifar10

# Import this PyPlot to visualize images
import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np

from sklearn.utils import shuffle

# Load dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
```


```python
# Print the shapes of training and testing set
print("X_train.shape =", X_train.shape, "Y_train.shape =", Y_train.shape)
print("X_test.shape =", X_test.shape, "Y_test.shape =", Y_test.shape)
```

    Output:
    X_train.shape = (50000, 32, 32, 3) Y_train.shape = (50000, 1)
    X_test.shape = (10000, 32, 32, 3) Y_test.shape = (10000, 1)


We can tell from the shapes that,

- **X_train** has 50000 training images, each 32 pixel wide, 32 pixel high, and 3 color channels
- **X_test** has 10000 testing images, each 32 pixel wide, 32 pixel high, and 3 color channels
- **Y_train** has 50000 labels
- **Y_test** has 10000 labels

Let us define constants for number of classes and its labels, to make the code more readable.


```python
NUM_CLASSES = 10
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]
```

Now, lets look at some random images from the training set. You can change the number of columns and rows to get more/less images.


```python
# show random images from training set
cols = 8 # Number of columns
rows = 4 # Number of rows

fig = plt.figure(figsize=(2 * cols, 2 * rows))

# Add subplot for each random image
for col in range(cols):
    for row in range(rows):
        random_index = np.random.randint(0, len(Y_train)) # Pick a random index for sampling the image
        ax = fig.add_subplot(rows, cols, col * rows + row + 1) # Add a sub-plot at (row, col)
        ax.grid(b=False) # Get rid of the grids
        ax.axis("off") # Get rid of the axis
        ax.imshow(X_train[random_index, :]) # Show random image
        ax.set_title(CIFAR10_CLASSES[Y_train[random_index][0]]) # Set title of the sub-plot
plt.show() # Show the image
```


![png](/assets/images/2019-05-12-CNN-on-CIFAR-10-using-TensorFlow/output_9_0.png)


## Prepare Training and Testing Data

Before defining the model and training the model, let us prepare the training and testing data.


```python
import tensorflow as tf
import numpy as np
print("TensorFlow's version is", tf.__version__)
print("Keras' version is", tf.keras.__version__)
```

    Output:
    TensorFlow's version is 1.13.1
    Keras' version is 2.2.4-tf


Normalize the inputs, to train the model faster and prevent exploding gradients.


```python
# Normalize training and testing pixel values
X_train_normalized = X_train / 255 - 0.5
X_test_normalized = X_test / 255 - 0.5
```

Convert the labels to one-hot coded vectors.


```python
# Convert class vectors to binary class matrices.
Y_train_coded = tf.keras.utils.to_categorical(Y_train, NUM_CLASSES)
Y_test_coded = tf.keras.utils.to_categorical(Y_test, NUM_CLASSES)
```

## Define Convolutional Neural Network Model

Next, let us define a model that takes images as input, and outputs class probabilities.

You can learn more about the implementation details
https://keras.io.

We will define following layers in the model:

- **Convolutional layer** which takes (32, 32, 3) shaped images as input, outputs 16 filters, and has a kernel size of (3, 3), with the same padding, and uses LeakyReLU as activation function
- **Convolutional layer** which takes (32, 32, 16) shaped tensor as input, outputs 32 filters, and has a kernel size of (3, 3), with the same padding, and uses LeakyReLU as activation function
- **Max Pool layer** with pool size of (2, 2), this outputs (16, 16, 16) tensor
- **Dropout layer** with the dropout rate of 0.25, to prevent overfitting
- **Convolutional layer** which takes (16, 16, 16) shaped tensor as input, outputs 32 filters, and has a kernel size of (3, 3), with the same padding, and uses LeakyReLU as activation function
- **Convolutional layer** which takes (16, 16, 32) shaped tensor as input, outputs 64 filters, and has a kernel size of (3, 3), with the same padding, and uses LeakyReLU as activation function
- **Max Pool layer** with pool size of (2, 2), this outputs (8, 8, 64) tensor
- **Dropout layer** with the dropout rate of 0.25, to prevent overfitting
- **Dense layer** which takes input from 8x8x64 neurons, and has 256 neurons
- **Dropout layer** with the dropout rate of 0.5, to prevent overfitting
- **Dense layer** with 10 neurons, and softmax activation, is the final layer

As you can see, all the layers use LeakyReLU activations, except the last layer. This is a pretty good choice most of the time, but you change these as well to play with other activations such as tanh, sigmoid, ReLU, etc.


```python
# import necessary building blocks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.layers import LeakyReLU

def make_model():
    """
    Define your model architecture here.
    Returns `Sequential` model.
    """
    
    model = Sequential()
    
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(LeakyReLU(0.1))
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(0.1))
    
    model.add(MaxPooling2D())
    
    model.add(Dropout(rate=0.25))
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(0.1))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(0.1))
    
    model.add(MaxPooling2D())
    
    model.add(Dropout(rate=0.25))
    
    model.add(Flatten())
    
    model.add(Dense(units=256))
    model.add(LeakyReLU(0.1))
    
    model.add(Dropout(rate=0.5))
    
    model.add(Dense(units=10))
    model.add(Activation("softmax"))
    
    return model
```


```python
# describe model
s = tf.keras.backend.clear_session()
model = make_model()
model.summary()
```

## Train your model

Next, we train the model that we defined above. We will use 0.005 as our initial learning rate, training batch size will be 64, we will train our model for 10 epochs. Feel free to change these hyperparameters, to dive deeper and know their effects. We use categorical cross entropy loss as our lass function and Adamax optimizer for convergence.


```python
INIT_LR = 5e-3  # initial learning rate
BATCH_SIZE = 64
EPOCHS = 10

s = tf.keras.backend.clear_session()  # clear default graph
# don't call K.set_learning_phase() !!! (otherwise will enable dropout in train/test simultaneously)
model = make_model()  # define our model

# prepare model for fitting (loss, optimizer, etc)
model.compile(
    loss='categorical_crossentropy',  # we train 10-way classification
    optimizer=tf.keras.optimizers.Adamax(lr=INIT_LR),  # for SGD
    metrics=['accuracy']  # report accuracy during training
)
```

We define a learning rate scheduler, which decays learning rate after each epoch.


```python
# scheduler of learning rate (decay with epochs)
def lr_scheduler(epoch):
    return INIT_LR * 0.9 ** epoch
```

We also define a class that handles callbacks from keras. It prints out the learning rate used in that epoch.


```python
# callback for printing of actual learning rate used by optimizer
class LrHistory(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        print("Learning rate:", tf.keras.backend.get_value(model.optimizer.lr))
```

Now, let us train our model on normalized X_train, **X_train_normalized**, and one-hot coded matrix, **Y_train_coded**. During training we will also keep validating on, **X_test_normalized** and **Y_train_coded**. In this way we can keep an eye on model performance.


```python
# fit model
history = model.fit(
    X_train_normalized, Y_train_coded,  # prepared data
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_scheduler), 
               LrHistory()],
    validation_data=(X_test_normalized, Y_test_coded),
    shuffle=True,
    verbose=1,
    initial_epoch=0
)
```


```python
def save_model(model):# serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


save_model(model)
```


## Evaluate the model

Now that we have trained our model, let us see how it performs.

Let us load the saved model from disk.


```python
def load_model():
    from tensorflow.keras.models import model_from_json
    
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    
    return loaded_model

model = load_model()
```

Let us look at the learning curve during the training of our model.


```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```


![png](/assets/images/2019-05-12-CNN-on-CIFAR-10-using-TensorFlow/output_32_0.png)


Let us predict the classes for each image in testing set.


```python
# make test predictions
Y_pred_test = model.predict_proba(X_test_normalized) # Predict probability of image belonging to a class, for each class
Y_pred_test_classes = np.argmax(Y_pred_test, axis=1) # Class with highest probability from predicted probabilities
Y_test_classes = np.argmax(Y_test_coded, axis=1) # Actual class
Y_pred_test_max_probas = np.max(Y_pred_test, axis=1) # Highest probability
```

Let us look at the confusion matrix to understand the performance of our model.


```python
# confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
plt.figure(figsize=(7, 6))
plt.title('Confusion matrix', fontsize=16)
plt.imshow(confusion_matrix(Y_test_classes, Y_pred_test_classes))
plt.xticks(np.arange(10), CIFAR10_CLASSES, rotation=45, fontsize=12)
plt.yticks(np.arange(10), CIFAR10_CLASSES, fontsize=12)
plt.colorbar()
plt.show()
print("Test accuracy:", accuracy_score(Y_test_classes, Y_pred_test_classes))
```


![png](/assets/images/2019-05-12-CNN-on-CIFAR-10-using-TensorFlow/output_36_0.png)


    Test accuracy: 0.7913


Test accuracy of ~ 80% isn't bad for such a simple model. Now, Let us look at some random predictions from our model.


```python
# inspect preditions
cols = 8
rows = 2
fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(Y_test))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid(b=False)
        ax.axis('off')
        ax.imshow(X_test[random_index, :])
        pred_label = CIFAR10_CLASSES[Y_pred_test_classes[random_index]]
        pred_proba = Y_pred_test_max_probas[random_index]
        true_label = CIFAR10_CLASSES[Y_test[random_index][0]]
        ax.set_title("pred: {}\nscore: {:.3}\ntrue: {}".format(
               pred_label, pred_proba, true_label
        ))
plt.show()
```


![png](/assets/images/2019-05-12-CNN-on-CIFAR-10-using-TensorFlow/output_38_0.png)


## Summary

In this tutorial, we discovered how to develop a convolutional neural network for CIFAR-10 classification from scratch using TensorFlow.

Specifically, we learned:

- How to load CIFAR-10 in your python program
- How to look at random images in the dataset
- How to define and train a model
- How to save the learnt weights of the model to disk
- How to predict clsses using the model


These topics will be covered later:

- How to improve your model
- How to thoroughly validate your model

This is a pretty good model (if it is among your first few), but people have achieved around 99% accuracy in this dataset. You can checkout other people's performance on this dataset [here](https://benchmarks.ai/cifar-10).

If you want to work on this model on your system, you can find th code [here](https://github.com/vbvsharma/build-cnn-on-cifar10-using-tensorflow).
