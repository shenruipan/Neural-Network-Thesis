import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np

tf.set_random_seed(123)
np.random.seed(123)

# Parameters
num_filter1 = 120
num_filter2 = 150
learning_rate = 0.1
num_training = 50000

# Default parameters
num_input = 784
dim_input = 28
num_output = 10
standard_deviation = 0.1
size_filter1 = 5
size_filter2 = 5
size_filter3 = 4
size_batch = 100

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = standard_deviation, seed = 123)
    return tf.Variable(initial, name= name )

def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name= name)

def conv2d_valid(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                          strides = [1, 2, 2, 1], padding = 'SAME')


x = tf.placeholder(tf.float32, [None, num_input])  # Input layer
y_ = tf.placeholder(tf.float32, [None, num_output])  # True values for output layer
x_image = tf.reshape(x, [-1, dim_input, dim_input, 1])     # Reshape input layer as matrix

# First convolutional layer
W_conv1 = weight_variable([size_filter1, size_filter1, 1, num_filter1], 'W_conv1')
b_conv1 = bias_variable([num_filter1], 'b_conv1')

h_conv1 = tf.nn.relu(conv2d_valid(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer
W_conv2 = weight_variable([size_filter2, size_filter2, num_filter1, num_filter2], 'W_conv2')
b_conv2 = bias_variable([num_filter2], 'b_conv2')

h_conv2 = tf.nn.relu(conv2d_valid(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Third convolutional layer
W_conv3 = weight_variable([size_filter3, size_filter3, num_filter2, num_output], 'W_conv3')
b_conv3 = bias_variable([num_output], 'b_conv3')

y_tensor = conv2d_valid(h_pool2, W_conv3) + b_conv3
y = tf.reshape(y_tensor, [-1, num_output])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Train
for i in range(num_training):
    batch_xs, batch_ys= mnist.train.next_batch(size_batch)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

saver = tf.train.Saver()
saver.save(sess, './LeNet1')