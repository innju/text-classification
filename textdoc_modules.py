# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:39:41 2022

This script consists of the classes and functions needed in textdoc_train.py
It can be connected with the training file through the existence of __init__.py
in the same folder as the training file.

@author: User
"""

#packages
import re
from tensorflow.keras.preprocessing.text import Tokenizer
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM,Embedding,Bidirectional,Dropout,Dense

#%% Classes and functions
class ExploratoryDataAnalysis():
    def __init__(self):
        pass
    
    def lower_split(self,data):
        for index,text in enumerate(data):
            data[index]= re.sub('[^a-zA-Z]',' ', text).lower().split()
            
        return data
    
    def sentiment_tokenizer(self,data,token_save_path,num_words=10000,
                            oov_token='<oov>'):
        # try with 10000 and see if the model able to perform well
        # tokenizer to vectorize the words
        tokenizer= Tokenizer(num_words=num_words, oov_token= oov_token)
        tokenizer.fit_on_texts(data)
        
        # to save the tokanizer
        token_json = tokenizer.to_json()
        
        with open(token_save_path,'w') as json_file:
            json.dump(token_json, json_file)
        
        # to observe the number of words
        word_index = tokenizer.word_index
        print(len(word_index)+1)
        
        # to vectorize the sequences of text
        data = tokenizer.texts_to_sequences(data)
            
        return data
    
    def sentiment_pad_sequence(self,data):
        return pad_sequences(data,maxlen=388,
                             padding='post',truncating= 'post')


class ModelCreation():
    def __init__(self):
        pass
    
    def lstm_layer(self,num_words,nb_categories,embedding_output=64,
                   nodes=32,dropout=0.2):
        model = Sequential()
        model.add(Embedding(num_words,embedding_output)) 
        model.add(Bidirectional(LSTM(nodes,return_sequences=True)))
        # return_sequences means keep 3D sequence input
        model.add(LSTM(nodes,return_sequences=True))
        model.add(Dropout(dropout)) # prevent overfitting
        model.add(Bidirectional(LSTM(nodes))) # by default it will use 'tanh'
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories,activation='softmax'))
        
        return model
