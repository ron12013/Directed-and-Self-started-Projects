""""
Here I download the wikipedia dump in my local machine, convert that into corpus and then into a zip file to run this program. Which means I can't run the entire program 
together at the same time. Also this is a comparitively slow program as it is bottle-necked due to usage of python to read , prepare the tokens and getting vocabulary size.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import io_ops
from tensorflow.python.lib.io import python_io

import zlib, base64
import collections
import math
import logging
import six
import sys
import os.path
import random
import argparse
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  
import tensorflow as tf
from gensim.corpora import WikiCorpus

tf.logging.set_verbosity(tf.logging.INFO)

#Trying to learn and then implement Custom Data Readers to optimize the code by significant margin
""""class SomeReader(io_ops.ReaderBase):
    def __init__(self, name = None):
        rr = gen_user_ops.some_reader(name = name)
        super(SomeReader, self).__init__(rr)
        
ops.NotDifferentiable("SomeReader")"""
        
             

if __name__ == '__main__':


# The function below changes the downloaded dump into corpus, where each line of the text file contains an article of Wikipedia
    def prepare_corpus():
        program = os.path.basename(sys.argv[0])
        logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
        logging.root.setLevel(level=tf.logging.INFO)
        logger.info("currently running %s" % ' '.join(sys.argv))

        # check and process input arguments
        if len(sys.argv) != 3:
            print("You should be using the folowing format: python this_filename.py enwiki.xxx.xml.bz2 filename.text")
            sys.exit(1)
        ip, op = sys.argv[1:3]
        space = " "
        i = 0

        output = open(op, 'w',encode ="utf-8")
        wiki = WikiCorpus(ip, lemmatize=False, dictionary ={}) # lemmatize is made false, in order to avoid a format which slows the process even more.
        for text in wiki.get_texts():
            
            output.write(b' '.join(text).decode('utf-8') + '\n')
            i = i + 1
            if (i % 10000 == 0):
                logger.info("Currently saved " + str(i) + " articles")

        output.close()
        logger.info("Total saving finished of : " + str(i) + " articles! ")
        return sys.argv[2]
        
    
    #filename = prepare_corpus()
    

    def read_data(filename):

      """"with tf.gfile.GFile(filename, "r") as f:
        data = f.read().replace("\n", "<eos>").split()

      return data"""
      with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
      return data
     

    def build_vocab(filename): # maybe a redundant function
      data = read_data(filename) 

      counter = collections.Counter(data)
      count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

      words, _ = list(zip(*count_pairs))
      word_to_id = dict(zip(words, range(len(words))))

      return word_to_id

    # Filename is the text file containing the corpus, extracted from the dump
    words = read_data("prb1.zip") #(prepare_corpus())
    print('Data size is :', len(words))


    vocabulary_size = len(build_vocab(filename))
    print(vocabulary_size) #For self clarification


    def build_dataset(words, vocabulary_size):
      count = [['UNK', -1]]
      count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
      dictionary = dict()
      for word, _ in count:
        dictionary[word] = len(dictionary)
      data = list()
      unk_count = 0
      for word in words:
        if word in dictionary:
          index = dictionary[word]
        else:
          index = 0  # dictionary['UNK']
          unk_count += 1
        data.append(index)
      count[0][1] = unk_count
      reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
      
      return data, count, dictionary, reverse_dictionary

    data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
    del words  # Hint to reduce memory.
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

    data_index = 0


  
    def generate_batch(batch_size, num_skips, skip_window):
      global data_index
      assert batch_size % num_skips == 0
      assert num_skips <= 2 * skip_window
      batch = np.ndarray(shape=(batch_size), dtype=np.int32)
      labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
      span = 2 * skip_window + 1  # [ skip_window target skip_window ]
      buffer = collections.deque(maxlen=span)
      
      for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
      for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        
        for j in range(num_skips):
          while target in targets_to_avoid:
            target = random.randint(0, span - 1)
          targets_to_avoid.append(target)
          batch[i * num_skips + j] = buffer[skip_window]
          labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
     
      data_index = (data_index + len(data) - span) % len(data)
      return batch, labels

    

   

    batch_size = 500
    embedding_size = 500  # Dimension of the embedding vector.
    skip_window = 2       # How many words to consider left and right.
    num_skips = 2         # How many times to reuse an input to generate a label.

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 26     # Random set of words to evaluate similarity on.
    valid_window = 200  # Only pick dev samples in the head of the distribution.
    #valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples, random.sample(range(1000,1000+valid_window), valid_size//2))
    num_sampled = 200    # Number of negative examples to sample.

    graph = tf.Graph()

    with graph.as_default(), tf.device('/cpu:0'):

        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)


            # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        print("The embed size would be: %s" %embed.get_shape().as_list()) # for self - clarrification

        softmax_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
        softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))


        loss = tf.reduce_mean(
          tf.nn.sampled_softmax_loss(weights=softmax_weights,
                         biases=softmax_biases,
                         labels=train_labels,
                         inputs=embed,
                         num_sampled=num_sampled,
                         num_classes=vocabulary_size)) # sampled softmax for large vocabulary size

        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss) #Used adagrad optimizer seemed to work better than Gradient Descent

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
          normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

      # Add variable initializer.
    init = tf.global_variables_initializer()


    num_steps = 1000001

    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
      tf.global_variables_initializer().run()
     

      average_loss = 0
      for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 5000 == 0:
          if step > 0:
            average_loss /= 5000
 
          print("Average loss at step ", step, ": ", average_loss)
          average_loss = 0

    
        if step % 100000 == 0:
          sim = similarity.eval()
          for i in xrange(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 15  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = "Nearest to %s:" % valid_word
            for k in xrange(top_k):
              close_word = reverse_dictionary[nearest[k]]
              log_str = "%s %s," % (log_str, close_word)
            print(log_str)
      final_embeddings = normalized_embeddings.eval()




    def plot_with_labels(low_dim_embs, labels, filename='problem1.png'):
      assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
      plt.figure(figsize=(20, 20))  #inches
      for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

      plt.savefig(filename)

    try:
      from sklearn.manifold import TSNE
      import matplotlib.pyplot as plt

      tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
      plot_only = 700
      low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
      labels = [reverse_dictionary[i] for i in xrange(plot_only)]
      plot_with_labels(low_dim_embs, labels)

    except ImportError:
      print("Install sklearn, matplotlib, and scipy to visualize embeddings.")
