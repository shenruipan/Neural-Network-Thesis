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
learning_rates = 0.1
num_training1 = 10000

# Default parameters
num_input = 784
dim_input = 28
standard_deviation = 0.1
size_filter1 = 5
size_batch = 100
iter_loss = 500

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = standard_deviation, seed = 123)
    return tf.Variable(initial, name= name )

def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name= name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

x = tf.placeholder(tf.float32, [None, num_input])  # Input layer

x_image = tf.reshape(x, [-1, dim_input, dim_input, 1])

# First convolutional autoencoder
W_conv1 = weight_variable([size_filter1, size_filter1, 1, num_filter1], 'W_conv1')
b_conv1 = bias_variable([num_filter1], 'b_conv1')
b_conv1_de = bias_variable([1], 'b_conv1_de')

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

x_image_recon = tf.nn.relu(conv2d(h_conv1, tf.transpose(W_conv1, perm=[1, 0, 3, 2])) + b_conv1_de)

loss1 = tf.reduce_mean(tf.square(x_image_recon - x_image))

train_step1 = tf.train.GradientDescentOptimizer(learning_rates).minimize(loss1, var_list=[W_conv1, b_conv1, b_conv1_de])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

loss1_data = np.zeros([int(num_training1/iter_loss), 1], dtype= float)
j = 0

# Train
for i in range(num_training1):
    batch_xs, batch_ys= mnist.train.next_batch(size_batch)
    sess.run(train_step1, feed_dict={x: batch_xs})
    if (i+1) % iter_loss == 0:
        loss1_data[j, 0] = np.asarray(sess.run([loss1], feed_dict={x: mnist.validation.images}))
        j = j + 1

plt.plot([(k+1)*iter_loss for k in range(int(num_training1/iter_loss))], loss1_data)

saver = tf.train.Saver()
saver.save(sess, './PanNet1_train1')
np.savetxt('loss.txt', loss1_data)
plt.savefig('PanNet1_train1.png', bbox_inches='tight')
