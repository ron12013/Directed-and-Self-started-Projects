# This code is an implementation of CNN with the MNIST hand written images 

import sys
import argparse
import numpy as nm
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import i_data

### Enter code to load in the MNIST images

 # SOLUTION:
mnist = i_data.read_data_sets("MNIST data/", one_hot = True)
sess = tf.InteractiveSession()

### Enter code to create placeholders for features and labels
#   Note that each MNSIT image of dimension 28 * 28, each vector corresponds to 784 * 1

# SOLUTION:
x = tf.placeholder([None, 784], tf.float32)
y = tf.placeholder(tf.float32, shape=[None,10])

# Before moving forward please note, the model of a CNN is usually in the following format
# convolution --> relu activation---> maxpooling --> repeat till we get a fully connected layer --> softmax

### Enter code to define functions which will return a tensorflow variable for weight and bias given the shape of the variable.
def weight_val(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_val(shape):
	initial = tf.constant(0.1,shape = shape)
	return tf.constant(initial)
	
sess.run(global_variable_initializer())


### Enter code to apply tensorflow's conv2d function with equal strides on all sides, i., write a function which returns conv2d on receving weights and inputs

# SOLUTION:

def convv(x,W):
	return tf.nn.conv2d(x,W,strides =[1,1,1,1], padding = 'SAME')

### Enter code to define the function for max pooling with strides = [1,2,2,1] and ksize = [1,2,2,1]	

#SOLUTION:

def maxp(x):
	return tf.nn.max_pool(x, strides =[1,2,2,1], ksize =[1,2,2,1], padding='SAME')

### Enter code to define the weights, bias and reshape the input layer
#   Note that the weights should be of 5 * 5 * 1, 32 as the output dimension 28 * 28 * 32, the input should be reshaped 28 * 28* 1 based on MNIST images

# SOLUTION:
 
W_c1 = weight_val([5,5,1,32])
b_c1 = bias_val([32])
x_i = tf.reshape(x, [-1,28,28,1])


###Enter code to get the input of the next layer, after applying convolution followed by relu
# Once you do that, apply max pooling on that layer 

# SOLUTION:

y_c1 = tf.nn.relu(convv(W_c1,x_i) + b_c1)
h_p1 = maxp(y_c1)

### Enter code to define the weights and bias for the next layer. Note that the weights should be of size 5 * 5 * 32, 64
#   Define the bias accordance to the weights.

# SOLUTION:

W_c2 = weight_val([5,5,32,64])
b_c2 = bias_val([64])

###Enter convolution , relu and maxpooling layer for the second layer of the network

# SOLUTION:

y_c2 = tf.nn.relu(convv(W_c2,h_p1) + b_c2)
h_p2 = maxp(y_c2)	

### Enter code to define the weights, and bias of the 3rd layer. Note that the weights would be of shape  7 * 7 * 64, 1024
# Try and calculate on your own to find why the dimensions of the weights is that.

# SOLUTION:

W_c3 = weight_val([7*7*64,1024])
b_c3 = bias_val([1024])

### Enter code to reshape the resulting layer after layer 2, and use relu on that to get the last convoluted layer
# Note: you may apply relu after the matrix multiplication which would give you the last convoluted layer

# SOLUTION:

x_f = tf.reshape(h_p2, [-1,7*7*64])
y_c3 = tf.nn.relu(tf.matmul(W_c3, h_p2)+ b_c3)

### Enter code to define a plaeholder which would be used when applying dropout. 
#  Apply dropout on the layer you just calculated above, you may use tensorflow's droput function.

# SOLUTION:

keep_prob = tf.placeholder(tf.float32)
h_fc_drop = tf.nn.dropout(y_c3,keep_prob)

### Enter code to apply the weights of the last layer, note this will depend on the number of labels present in the dataset
# Define your bias accordingly.

# SOLUTION:

W_fc= weight_val([1024,10])
b_fc = bias_val([10])


###Enter code to get the logits for the FC layer

# SOLUTION:

y_conv = tf.matmul(W_fc,h_fc_drop) + b_fc


### Enter code to apply the softmax cross entropy loss with logits, to get the cross entropy loss
# Once you have that loss, try to minimize it using tensorflow's optimization function ( gradient descent or adam's optimizer)

# SOLUTION:
cross_e = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_e)

global_variable_initializer().run()

for i in range(2000):
	batch = mnist.train.next_batch(50)
	correct_pred = tf.equal(tf.argmax(y_,1), tf.argmax(y_conv,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	if (i%100 ==0)
		train_a = accuracy.eval(feed_dict = {x: batch[0], y: batch[1], keep_prob :1.0})
		print("step %d training accurancy %g", (i,train_a))
	train_step.run(feed_dict{x:batch[0], y_:batch[1], keep_prob:0.5})
print("Test accuracy:%g",% accuracy.eval(feed_dict = {x:mnist.test.images, y_:mnsit.test.labels, keep_prob:1.0}))
