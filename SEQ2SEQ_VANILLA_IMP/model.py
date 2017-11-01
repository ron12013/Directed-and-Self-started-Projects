import tensorflow as tf
import random
import numpy as np

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

     ###   
    def lstm_cell(self):
      
        return tf.contrib.rnn.GRUCell(self.hidden_features)

        
    def build_model(self):
        ### Enter code for declaring placeholders named as defined above.
        # Note: X should be of size of the number of time steps by the input size
        #       hidden states should be a placeholder of dimension of the number of hidden features
        #       Y should be a placeholder of dimension of the number of output classes ###
        
        #SOLUTION:
        
        self.X = tf.placeholder( "float",[None, self.time_steps, self.input_size])
        hidden_state = tf.placeholder("float", [None, self.hidden_features], name="Hidden")
        self.Y = tf.placeholder("float", [None, self.output_classes],  name="Output")

        ### Enter code to define tensorflow variables for the hidden and outside weights and biases
        #   Note: The variables should be values of random distribution, where the hidden weights and biases
        #   should have dimension based on the hidden features. The outside weights and biases are based on 
        #  the hidden feature and output classes ###
        
        # SOLUTION:
        W_hidden = tf.Variable(tf.random_normal([self.input_size,self.hidden_features]))
        W_out = tf.Variable(tf.random_normal([self.hidden_features,self.output_classes]))
        b_hidden = tf.Variable(tf.random_normal([self.hidden_features]))
        b_out = tf.Variable(tf.random_normal([self.output_classes]))

        # The Formula for the Model
        # Reshape the input and split it according to the number of time steps , you an use tensorflow's split function.
        
        # SOLUTION:
        
        input_ = tf.reshape(self.X, [-1, self.input_size])
        input_2 = tf.split(input_, self.time_steps,0)
        
        ## Enter code to define the RNN cell which will be used in the autoencoder. You may use tensorflow's MultiRNN function
        # which could either call a standard RNN (of tensorflow) or a GRU for each layer, for the total number of layers ###
        
        # SOLUTION:
        
        cells = tf.contrib.rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.num_layers)])
        
        ### Enter code to define a hidden state based on size, then use it on a basic sequence to sequence function.
        #  You may use tensorflow seq2seq function with basic rnn.
        # Note that the tensorflow's legacy_seq2seq takes in the input the kind of cells you define and the hidden state.
        
        #SOLUTION:
        hidden_state = cells.zero_state(self.batch_size, tf.float32)
        outputs, state = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(input_2, hidden_state, cells)
        
        ### Enter code to define the hypothesis or logits using he outputs from the seq2seq autoencoder, 
         # using the outside weight and biases defined above to get the logits. ###
            
         # SOLUTION:   
        
        hypothesis = tf.matmul(outputs[-1], W_out) + b_out

        ### Enter code to calculate the cost of the model, use softmax cross entropy function of tensorflow
        # Use that cost to try and optimize the loss.
        # Use gradient descent or adam's optimizer of tensorflow to implement a loss minimization function of logits
        # You can use tensorflow's inbuilt optimization models to get the results.
        
        # SOLUTION:
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels = self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)

        # Enter code to calculate the accuracy, you may use tensorflow's argmax and cast function, in order to get the maximum 
        # value in each row. 
        
        # SOLUTION:
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(self.Y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.acc_summary = tf.summary.scalar("Accuracy", self.accuracy)
