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

Apart from these there were many other sources of learning inluding the official website of Tensorflow and Keras.


# Exercise 1: Getting familiar with Tensorlfow
The first Exercise was going through tensorflow and getting myself familiar with it. I followed the official tensorflow page where there were multiple exercises for beginners which would get them amiliar with concepts like, placeholders, tensorflow session and so on.
https://www.tensorflow.org/get_started/

# 1 a - Logistic Regression with MNIST handwritten digits
This exercise is about making an image classifier using logistic regression, we classify hand written digits from 0-9 (MNIST dataset)
The idea is getting familiar with placeholders and variables which hold the required tensors to get the logits, followed by finding the cross entropy loss which helps you get the batch. That is followed by using gradient descent optimization which will help you reduce the loss.Then we launch a tensorflow session which which will help to calculate loss one batch at a time, after which we use the trained model with the test data to get the final accuracy.
You will find the solution in the file  mnist_lr.py
# 1 b - 

# Exercise 2: Word2vec (skipgram) model  on latest wikipedia dump
This exercise involved downloading the latest wikipedia dump, and extracting a corpus out of it. Once you have that corpus you can use that to get word2vec embeddings of that corpus. 

# Exercise 3: Sequence2Sequence autoencoder using the wikipedia dump
This is an exercise involves taking the existing wikipedia dump and using it on a seq2seq autoencoder. The aim is to take a sentence train the model to learn the minimal representation of a sentence using the encoder which as we know converts the input sentence into a fixed representation. 
