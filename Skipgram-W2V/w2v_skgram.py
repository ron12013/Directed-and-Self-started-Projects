#This is the solution to word2vec skipgram model on Wikipedia dump.

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

batch_size = 500
embedding_size = 500  # Dimension of the embedding vector.
skip_window = 2       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
num_sampled = 200    # Number of negative examples to sample.
valid_size = 26     # Random set of words to evaluate similarity on.
valid_window = 200  # Only pick dev samples in the head of the distribution.
tf.logging.set_verbosity(tf.logging.INFO)
             

if __name__ == '__main__':

# The function below is a script which changes the dump into Corpus, in a format where each line of the resulting text file 
# will contain one wikiepdia article. It reads in the .bz2 file along with the text file where it will write to.
# THIS IS AN OPTIONAL FUNCTION, YOU CAN IMPLEMENT IN A SEPERATE FILE, ITS MAIN IDEA IS TO GET THE CORPUS.
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

      with tf.gfile.GFile(filename, "r") as f:
        data = f.read().replace("\n", "<eos>").split()

      return data
      """with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
      return data"""
     


    # Filename is the text file containing the corpus, extracted from the dump. 
    #The idea here is to get the size of the corpus by calling the read_data function.
   
    words = read_data(prepare_corpus())
    print('Data size is :', len(words))


    vocabulary_size = len(build_vocab(filename))
    print(vocabulary_size) #For self clarification

#This function is the dataset buider, where each word is tokenized for better access. 
# This function would be later called when we actually start calculating the embeddings

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

# This function deals with generating batches which would be used
  
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

    

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
   
    #valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples, random.sample(range(1000,1000+valid_window), valid_size//2))


    graph = tf.Graph()

    with graph.as_default(), tf.device('/cpu:0'):

       #The line below defines
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
