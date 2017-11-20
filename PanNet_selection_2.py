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
num_filter1 = 4
num_filter2 = 12
learning_rates = 0.1
num_training1 = 4000
num_training2 = 10000

# Default parameters
num_input = 784
dim_input = 28
standard_deviation = 0.1
size_filter1 = 5
size_filter2 = 5
size_batch = 100
iter_loss = 250

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = standard_deviation, seed = 123)
    return tf.Variable(initial, name= name )

def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name= name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                          strides = [1, 2, 2, 1], padding = 'SAME')

x = tf.placeholder(tf.float32, [None, num_input])  # Input layer
x_image = tf.reshape(x, [-1, dim_input, dim_input, 1])

# First convolutional autoencoder
W_conv1 = weight_variable([size_filter1, size_filter1, 1, num_filter1], 'W_conv1')
b_conv1 = bias_variable([num_filter1], 'b_conv1')
b_conv1_de = bias_variable([1], 'b_conv1_de')

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

x_image_recon = tf.nn.relu(conv2d(h_conv1, tf.transpose(W_conv1, perm=[1, 0, 3, 2])) + b_conv1_de)

# Second convolutional autoencoder
W_conv2 = weight_variable([size_filter2, size_filter2, num_filter1, num_filter2], 'W_conv2')
b_conv2 = bias_variable([num_filter2], 'b_conv2')
b_conv2_de = bias_variable([num_filter1], 'b_conv2_de')

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)

h_pool1_recon= tf.nn.relu(conv2d(h_conv2, tf.transpose(W_conv2, perm=[1, 0, 3, 2]))+b_conv2_de)

loss1 = tf.reduce_mean(tf.square(x_image_recon - x_image))
loss2 = tf.reduce_mean(tf.square(h_pool1_recon - h_pool1))

train_step1 = tf.train.GradientDescentOptimizer(learning_rates).minimize(loss1, var_list=[W_conv1, b_conv1, b_conv1_de])
train_step2 = tf.train.GradientDescentOptimizer(learning_rates).minimize(loss2, var_list = [W_conv2, b_conv2, b_conv2_de])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

loss2_data = np.zeros([int(num_training2/iter_loss), 1], dtype= float)
j = 0

# Train
for _ in range(num_training1):
    batch_xs, batch_ys= mnist.train.next_batch(size_batch)
    sess.run(train_step1, feed_dict={x: batch_xs})

for i in range(num_training2):
    batch_xs, batch_ys= mnist.train.next_batch(size_batch)
    sess.run(train_step2, feed_dict={x: batch_xs})
    if i % iter_loss == 0:
        loss2_data[j, 0] = np.asarray(sess.run([loss2], feed_dict={x: mnist.validation.images}))
        j = j + 1

plt.plot([(k*iter_loss+1) for k in range(int(num_training2/iter_loss))], loss2_data)
plt.xlabel('Number of training')
plt.ylabel('Autoencoder loss')
plt.title('Autoencoder loss for the second convolutional autoencoder')

saver = tf.train.Saver()
saver.save(sess, './PanNet_train2')
np.savetxt('loss2.txt', loss2_data)
plt.savefig('PanNet_train2.png', bbox_inches='tight')