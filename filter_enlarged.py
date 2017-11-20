import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np
from pylab import *

#pixels1 = mnist.train.images[1, :] # convert tensor to array
#pixels1 = np.reshape(pixels1, [28, 28])
#plt.imshow(pixels1, cmap = 'winter')
#plt.axis('off')
#plt.figure(1)
#plt.colorbar()
#plt.show()

num_filter1 = 120
num_input = 784
dim_input = 28
size_filter1 = 5

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('C:/Users/Veblen/mystuff/.idea/filter/PanNet-enlarged/PanNet.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    W_conv1 = sess.run("W_conv1:0")

x = tf.placeholder(tf.float32, [None, num_input])  # Input layer
x_image = tf.reshape(x, [-1, dim_input, dim_input, 1])     # Reshape input layer as matrix

filter = tf.placeholder(tf.float32, [size_filter1, size_filter1, 1, num_filter1])

output = tf.nn.relu(conv2d(x_image, filter))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

result = np.sum(sess.run(output, feed_dict={x:mnist.train.images[0:1000,:],
                                     filter: W_conv1}), axis = 0)

for i in range(120):
    pixels = result[:,:, i] # convert tensor to array
    plt.subplot(12, 10, i+1)
    plt.imshow(pixels, cmap = 'winter')
    plt.axis('off')
plt.figure(1)
plt.show()

for i in range(120):
    pixels = W_conv1[:,:,0, i] # convert tensor to array
    plt.subplot(12, 10, i+1)
    plt.imshow(pixels, cmap = 'winter')
    plt.axis('off')
plt.figure(2)
plt.show()
