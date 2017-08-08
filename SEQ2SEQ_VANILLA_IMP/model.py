import tensorflow as tf
import random
import numpy as np
#Files contain some borrowed code

class Sequence2Sequence():

    def __init__(self, args):
        self.args = args
        # Parameters for the model
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.time_steps = args.time_steps
        self.hidden_features = args.hidden_features
        self.output_classes = args.output_classes
        self.num_layers = args.num_layers

        # The below are made in build_model and are needed for training.
        self.X = None
        self.Y = None
        self.hypothesis_index = None
        self.optimizer = None
        self.accuracy = None
        self.acc_summary = None

        #  call to build the model
        self.build_model()

        
    def lstm_cell(self):
      
        return tf.contrib.rnn.GRUCell(self.hidden_features)

        
    def build_model(self):
        # Placeholders for our input data, hidden layer, and y values
        self.X = tf.placeholder( "float",[None, self.time_steps, self.input_size])
        hidden_state = tf.placeholder("float", [None, self.hidden_features], name="Hidden")
        self.Y = tf.placeholder("float", [None, self.output_classes],  name="Output")

        # Weights and Biases for hidden layer and output layer
        W_hidden = tf.Variable(tf.random_normal([self.input_size,self.hidden_features]))
        W_out = tf.Variable(tf.random_normal([self.hidden_features,self.output_classes]))
        b_hidden = tf.Variable(tf.random_normal([self.hidden_features]))
        b_out = tf.Variable(tf.random_normal([self.output_classes]))

        # The Formula for the Model
        input_ = tf.reshape(self.X, [-1, self.input_size])

        
        input_2 = tf.split(input_, self.time_steps,0)
        
        cells = tf.contrib.rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.num_layers)])
        
        hidden_state = cells.zero_state(self.batch_size, tf.float32)
        outputs, state = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(input_2, hidden_state, cells)
        hypothesis = tf.matmul(outputs[-1], W_out) + b_out
        self.hypothesis_index = tf.argmax(hypothesis,1)

        # Define our cost and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels = self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)

        # Define our model evaluator
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(self.Y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.acc_summary = tf.summary.scalar("Accuracy", self.accuracy)
