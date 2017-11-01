# Independent Study report
This repository includes details of the exercises that I did and study materials I used for my Independent Study in NLP which I am taking with Professor Anna Rumshisky. The exercises uses various Machine Learning/Deep Learning libraries in python. 

# Study Materials:
Some of the study materials that I went through in order to make myself familiar with NLP and its nuances
Andrej Karpathy's Blog:
http://karpathy.github.io/2015/05/21/rnn-effectiveness/

Richard Socher's Stanford NLP class lectures:
https://www.youtube.com/watch?v=Qy0oEkCZkBI&list=PLlJy-eBtNFt4CSVWYqscHDdP58M3zFHIG

Blog on sequence2sequence models:
https://indico.io/blog/sequence-modeling-neuralnets-part1/

Yoshua Bengio's lecture on Recurrent Neural Networks:
https://www.youtube.com/watch?v=xK-bzjIQkmM

Lex Fridman's MIT lecture on RNNs:
https://www.youtube.com/watch?v=nFTQ7kHQWtc&t=138s

Apart from these there were many other sources of learning inluding the official website of Tensorflow, Numpy, Keras and so on...


# Exercise 1: Getting familiar with Tensorflow
The first Exercise was going through tensorflow and getting myself familiar with it. I followed the official tensorflow page where there were multiple exercises for beginners which would get them familiar with concepts like, placeholders, tensorflow session and so on.

https://www.tensorflow.org/get_started/

# 1 a - Logistic Regression with MNIST handwritten digits images
This exercise is about making an image classifier using logistic regression, we classify hand written digits from 0-9 (MNIST dataset)
The idea is getting familiar with placeholders and variables which hold the required tensors to get the logits, followed by finding the cross entropy loss which helps you get the batch. 

That is followed by using gradient descent optimization which will help you reduce the loss.Then we launch a tensorflow session which which will help to calculate loss one batch at a time, after which we use the trained model with the test data to get the final accuracy.

You will find the solution in the file  mnist_lr.py

# 1 b - Convolutional Neural Networks with MNIST handwritten digits images
This is the second exercise in order to get yourself familiar with tensorflow. Here we implement a  CNN (Convolutional Neural Networks)
to classify images. In a CNN we build one layer on top of the other. Tensorflow's graph structure is benificial for a such an implementation. Convolutional Neural Nets are comparitively easy to understand compared to LSTM or GRU.

There are many features of Tensorflow which will help you implement this such as, tensorflow's function for pooling (max pooling), relu and fully connected layer.

The file cnn_mnist.py contains the solution.

# Exercise 2: Word2vec (skipgram) model on latest Wikipedia dump
This exercise involve downloading the latest wikipedia dump, and extracting a corpus out of it. Once you have that corpus you can use that to get word2vec embeddings of that corpus. 
So one of the most important taks in NLP is getting good word vectors (i.e., reprentations of words). We tokenize words so that it becomes easier to vectorize it later. Vectorizing means that each word token is representated as a vector of size of 300. We call this as the embedding vector. 

According to Richard Socher, as he mentions during one of his Stanford lectures that a good embedding representatons will capture some co-relation among words (The famous king and queen example. Check out the lectures more, link present above)
This exercise is implementing the skipgram model of word2vec where you use the center word to predict the surrounding words (called target words) 

The solution is present in the file w2v_skgram.py
The solution includes the data processing section, which you may include in a seperate file and need not know in detail as a beginner.


# Exercise 3: Sequence to Sequence autoencoder implementation
We know that RNN (Recurrent Neural Networks) and LSTM (Long Short Term Memory) networks are used extensively in the field of NLP and Computer Vison. This is an exercise involves taking a corpus and using it on a sequence to sequence autoencoder. The aim is to take a sentence train the model to learn the minimal representation of a sentence using the encoder which as we know converts the input sentence into a fixed representation. 

In order to implement, you could use tensorflow's multi RNN cell function, using a GRU cell for each layer or you may use the standard RNN cell units of tensorflow. Now you use the tensorflow's RNN sequence to sequence model giving it the cell type (from the Multi RNN (GRU)), encoder inputs and decoder inputs as the parameters. This model first runs an RNN to encode the inputs of the encoer and then runs decoder which is initialized with the last encoder state on the inputs of the decoder. The decoder and the encoder share the same RNN cell type but they do not share the same set of parameters.

The solution is contained in a folder Sequence_to_sequence_vanilla_implementation.
The files data_utils.py and train.py deals with reading in the data and running the tensorflow session.
model.py contains the solution.
