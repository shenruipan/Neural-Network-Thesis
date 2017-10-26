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
iter_accuracy = 1000

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
correct_prediction=tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

accuracy_test = np.zeros([100, 1], dtype = float)
accuracy_test_data = np.zeros([int(num_training/iter_accuracy), 1], dtype= float)
j = 0

# Train
for i in range(num_training):
    batch_xs, batch_ys= mnist.train.next_batch(size_batch)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
    if (i + 1) % iter_accuracy == 0:
        for k in range(100):
            accuracy_test[k, 0] = sess.run(accuracy, feed_dict={x: mnist.test.images[k * 100:(k + 1) * 100, :],
                                                            y_: mnist.test.labels[k * 100:(k + 1) * 100, :]})
        accuracy_test_data[j, 0] = sum(accuracy_test) / 100
        j = j + 1

plt.figure(1)
plt.plot([(k+1)*iter_accuracy for k in range(int(num_training/iter_accuracy))], accuracy_test_data)
plt.xlabel('Number of training')
plt.ylabel('Accuracy')
plt.title('Accuracy for LeNet1-enlarge')
plt.savefig('LeNet1-enlarge_acc.png', bbox_inches='tight')

saver = tf.train.Saver()
saver.save(sess, './LeNet1-enlarge')
np.savetxt('LeNet1-enlarge_test_accuracy.txt',accuracy_test_data )