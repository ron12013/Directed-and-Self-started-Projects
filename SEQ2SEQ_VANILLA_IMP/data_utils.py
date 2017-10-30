import re
import numpy as np
import collections
import os
import tensorflow as tf


def get_data(end=100, data_dir=""):

    with open('source.txt','r',encoding ='UTF8') as s:
        source = ''
      
        i = 0
        while i < 100:
            source_line = s.readline()
       
            if not source_line: #or not target_line:
                break
            source_line = re.sub(' +',' ',source_line)
            source_line = re.sub('\n',' ',source_line)
        
            source += source_line
            # target += target_line
            i+=1
        source = list(source)
        
        print(len(source))
        # print(len(target))
        # if len(source) != len(target):
            # raise ValueError("The length of source and target does not match")
        for i in range(len(source)):
            source[i] = ord(source[i])
            # target[i] = ord(target[i])
        source = np.asarray(source).astype(np.float32,copy=False)
        # target = np.asarray(target).astype(np.float32,copy=False)
        return source

""""def get_data(data_dir=""):

    #def _read_words('text8.txt'):
        with tf.gfile.GFile('text8.txt', "r") as f:
            data =  f.read().replace("\n", "<eos>").split()


        #def _build_vocab('text8.txt'):
        #data = _read_words('text8.txt')

        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))

        #return word_to_id


        #def _file_to_word_ids('text8.txt', word_to_id):
        #data = _read_words('text8.txt')
        print(len(word_to_id))
        return [word_to_id[word] for word in data if word in word_to_id] """
