"""
    Last semester, I implemented a from-scratch neural network to classify handwritten digits from the MNIST dataset.
    This semester, my goal is to become fluent with the machine learning framework TensorFlow. I hope to have at least 3
    different machine learning projects up on my github using the framework by the time I start my internship at Cerner.

    The first project will be a simple one to get a feel for the framework and start to really understand how to
    manipulate and interact with the code rather than just stealing others' code. This project will be a tensorflow
    implementation of the project I worked on last semester, classifying handwritten digits from the MNIST dataset. I
    will be using a tutorial by Justin Francis from a little over a year ago:

    https://www.oreilly.com/learning/not-another-mnist-tutorial-with-tensorflow

    I chose this one because it is visually appealing and interactive, so I really learn rather than just copying code.
    So without further ado....

    TENSORFLOW MNIST -- by Eli Anderson
"""

# import  the MNIST dataset (tensorflow has it conveniently built-in)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# one_hot encoding: 5 = 0000010000, 2 = 0010000000, etc. This is how the from-scratch model worked as well

# import other necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

# define a couple functions that will assign the amount of training and test data loaded from the dataset

def train_size(num) :
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    print('Total training images in dataset = ' + str(mnist.train.images.shape))

    # x_train = the pixel data itself
    x_train = mnist.train.images[:num,:]
    print('x_train examples loaded = ' + str(x_train.shape))

    # y_train = the image labels
    y_train = mnist.train.labels[:num,:]
    print('y_train examples loaded = ' + str(y_train.shape))
    print('\n')
    return x_train, y_train

def test_size(num) :
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    print('Total testing images in dataset = ' + str(mnist.test.images.shape))
    x_test = mnist.test.images[:num, :]
    print('x_test examples loaded = ' + str(x_test.shape))
    y_test = mnist.test.labels[:num, :]
    print('y_train examples loaded = ' + str(y_test.shape))
    print('\n')
    return x_test, y_test

# some simple functions for resizing & displaying the data

def display_digit(num) :
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    print('Digit\'s one_hot encoding: ' + str(y_train[num]))

    # since labels are stored in one_hot lists, use argmax to return the index, which is the actual label
    label = y_train[num].argmax(axis=0)
    # images are 28x28 pixels
    image = x_train[num].reshape([28,28])
    plt.title('Example: #' + str(num) + ', Label: ' + str(label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

# time to build and train the model
x_train, y_train = train_size(10000)
# some arbitrary digit
display_digit(2753)

# the next step is to actually use tensorflow to create a flow chart (https://www.tensorflow.org/programmers_guide/graph_viz),
# which will later be fed the data

# define the session
sess = tf.Session()

# define a placeholder -- a variable used to feed data into (i.e., the input layer). Its shape and type need to be
# matched exactly. Let's define x as the placeholder to feed the x_train data into:
x = tf.placeholder(tf.float32, shape=[None, 784])
#  the input layer can be fed as many examples as you want(None) of 784-sized values

# define y_ as the placeholder to feed the y_train data into. This will just be used later to compare our predictions to
# ground truths
y_ = tf.placeholder(tf.float32, shape=[None, 10])
