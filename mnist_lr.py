import tensorflow as tf
import sys
import argparse
import numpy as nm
from tensorflow.examples.tutorials.mnist import input_data
FLAGS = "NONE"

n_epochs = 1000
learning_rate = 0.5


def main(_):
	#The line below reads in the MNSIT data into the folder MNIST_data/
	mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

	#The line below creates the placeholders for the label and features
	#We have images of digits from 0-9 and each image here is of 784 by 1 tensor as MNIST digits are 28*28

	x = tf.placeholder(tf.float32, [None, 784])
	y_ = tf.placeholder(tf.float32,[None,10])
	
	#The lines below create defines the variable weights and bias, weights depend on the the input tensors
	#i.e., X and the b will depend on y as we need them to be the same size
	W = tf.Variable(tf.zeros([784,10]))
	b = tf.Variable(tf.zeros([10]))

	#The matrix multiplication takes place here where we get the logits and use it later in softmax layer

	y = tf.matmul(x,W) + b

	
	#This step is to get the cross entropy loss with the softmax of logits,
	#We use the tf.nn.softmax_cross_entropy_with_logits for that, then we use tf.reduce_mean to
	#get the mean loss of the batch
	cross_e = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels =y_,logits =y))
	
	
	#This step is the gradient descent optimizer for loss minimization with appropriate learning rate
	
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_e)
	
	#The line below launches the tensorflow session which will iteratively help us calculate the final accuracy
	
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	
	for _ in range(n_epochs):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # This step is the test the trained model
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images,
										  y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
