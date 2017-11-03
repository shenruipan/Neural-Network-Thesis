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
learning_rates = 0.01
num_training1 = 2000
num_training2 = 2000
num_training3 = 50000
iter_accuracy = 1000

# Default parameters
num_input = 784
dim_input = 28
num_output = 10
standard_deviation = 0.1
size_filter1 = 5
size_filter2 = 5
size_filter3 = 7
size_batch = 100

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = standard_deviation, seed = 123)
    return tf.Variable(initial, name= name )

def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name= name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def conv2d_valid(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                          strides = [1, 2, 2, 1], padding = 'SAME')

x = tf.placeholder(tf.float32, [None, num_input])  # Input layer
y_ = tf.placeholder(tf.float32, [None, num_output])  # True values for output layer

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
h_pool2 = max_pool_2x2(h_conv2)

h_pool1_recon= tf.nn.relu(conv2d(h_conv2, tf.transpose(W_conv2, perm=[1, 0, 3, 2]))+b_conv2_de)

# Third convolutional layer
W_conv3 = weight_variable([size_filter3, size_filter3, num_filter2, num_output], 'W_conv3')
b_conv3 = bias_variable([num_output], 'b_conv3')

y_tensor = conv2d_valid(h_pool2, W_conv3) + b_conv3
y = tf.reshape(y_tensor, [-1, num_output])

loss1 = tf.reduce_mean(tf.square(x_image_recon - x_image))
loss2 = tf.reduce_mean(tf.square(h_pool1_recon - h_pool1))
loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
correct_prediction=tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step1 = tf.train.GradientDescentOptimizer(learning_rates).minimize(loss1, var_list=[W_conv1, b_conv1, b_conv1_de])
train_step2 = tf.train.GradientDescentOptimizer(learning_rates).minimize(loss2, var_list = [W_conv2, b_conv2, b_conv2_de])
train_step3 = tf.train.GradientDescentOptimizer(learning_rates).minimize(loss3, var_list = [W_conv3, b_conv3])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Train
for _ in range(num_training1):
    batch_xs, batch_ys= mnist.train.next_batch(size_batch)
    sess.run(train_step1, feed_dict={x: batch_xs})

for _ in range(num_training2):
    batch_xs, batch_ys= mnist.train.next_batch(size_batch)
    sess.run(train_step2, feed_dict={x: batch_xs})

accuracy_test = np.zeros([100, 1], dtype = float)
accuracy_test_data = np.zeros([int(num_training3/iter_accuracy), 1], dtype= float)
j = 0

for i in range(num_training3):
    batch_xs, batch_ys= mnist.train.next_batch(size_batch)
    sess.run(train_step3, feed_dict={x: batch_xs, y_:batch_ys})
    if (i + 1) % iter_accuracy == 0:
        for k in range(100):
            accuracy_test[k, 0] = sess.run(accuracy, feed_dict={x: mnist.test.images[k * 100:(k + 1) * 100, :],
                                                               y_: mnist.test.labels[k * 100:(k + 1) * 100, :]})
        accuracy_test_data[j, 0] = sum(accuracy_test)/100
        j = j + 1

plt.figure(1)
plt.plot([(k+1)*iter_accuracy for k in range(int(num_training3/iter_accuracy))], accuracy_test_data)
plt.xlabel('Number of training batches')
plt.ylabel('Accuracy')
plt.title('Accuracy for PanNet-enlarged')
plt.savefig('PanNet-enlarged_acc.png', bbox_inches='tight')

saver = tf.train.Saver()
saver.save(sess, './PanNet-enlarged')
np.savetxt('test_accuracy_0_0_1_1000.txt',accuracy_test_data )