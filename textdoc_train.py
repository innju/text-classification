# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:26:59 2022

This script is used to categorize unseen articles into 5 categories
namely Sport,Tech,Business,Entertainment and Politics.

@author: User
"""

# Packages
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
import datetime
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from textdoc_modules import ExploratoryDataAnalysis,ModelCreation


#%% Static code
URL= 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
TOKENIZER_JSON_PATH = os.path.join(os.getcwd(),'tokenizer_sentiment.json')
LOG_PATH = os.path.join(os.getcwd(),'log_sentiment_analysis') #for tensorboard use
log_dir = os.path.join(LOG_PATH,datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
MODEL_SAVE_PATH= os.path.join(os.getcwd(),'sentiment_analysis.h5') #save model
OHE_SAVE_PATH = os.path.join(os.getcwd(),'ohe.pkl')


#%% EDA
#1) Load data
df = pd.read_csv(URL) # load data from URL
df.head()
df.shape #(2225,2)
#check the count of target variable
df['category'].value_counts()
# sport            511
# business         510
# politics         417
# tech             401
# entertainment    386
# The text document for analysis consist of most info for sport and business

category= df['category'] #target
review = df['text'] #texts


#2) Data inspection
# view samples of data
print(review[10])
print(review[500])
print(review[100])
# consists of punctuations,spacing,numerical data
# in lowercase and uppercase


#3) Data cleaning
# to convert to lowercase and split it
# remove the punctuations,spacing,numerical data,left only alphabet in review
eda = ExploratoryDataAnalysis()
review = eda.lower_split(review)


# recheck data
print(review[10])
print(review[500])
print(review[100])
# all in lowercase and splitted to word


#4) Feature selection
# Not needed


#5) Data preprocessing
# data vectorization
review = eda.sentiment_tokenizer(review, TOKENIZER_JSON_PATH)
# vocabulary size identified is 27908


# to check the number of words inside the list
temp = ([np.shape(i) for i in review])
np.mean(temp) # mean of words--->386.27
# ensure all sequences have the same length
# hence, maxlen in pad sequence=388
review = eda.sentiment_pad_sequence(review)
# padded zero to the end after 388th


# one hot encoding for target
ohe= OneHotEncoder(sparse=False)
category_encoded = ohe.fit_transform(np.expand_dims(category,axis=-1))
# save one hot encoder to pickle file
ohefile='ohe.pkl'
pickle.dump(ohe,open(ohefile,'wb'))

# to calculate the total number of categories
print(len(np.unique(category))) # 5 categories

# by comparing the category encoded and category, know that
# [0. 0. 1. 0. 0.] refer to politics
# [1. 0. 0. 0. 0.] refer to business
# [0. 0. 0. 0. 1.] refer to tech
# [0. 0. 0. 1. 0.] refer to Sport
# [0. 1. 0. 0. 0.] refer to entertainment
# another option to check the label encoder and the category represented
# print (y_train[i])
# print(one_hot_encoder.inverse_transform(np.expand_dims(y_train[i],axis=0)))

# X= review(features), y= category (target)
X_train, X_test, y_train, y_test= train_test_split(review,
                                                   category_encoded,
                                                   test_size=0.3,
                                                   random_state=123)
# expand dimension
X_train= np.expand_dims(X_train, axis=-1)
X_test= np.expand_dims(X_test, axis=-1)


#%% Model creation

mc= ModelCreation()
num_words = 10000 # no.of unique words
nb_categories= 5
# added bidirectional to fasten the process
# embedding approach (faster time to process during deployment)
# increase the accuracy as well
model = mc.lstm_layer(num_words,nb_categories)
model.summary()
# to view the architecture of the model
plot_model(model)

#%% Performance evaluation
# using tensorboard to view the performance
tensorboard_callback = TensorBoard(log_dir= log_dir, histogram_freq=1)

model.compile(optimizer='adam',
              loss='categorical_crossentropy', #categorical problem
              metrics='acc')

model.fit(X_train,y_train,epochs=30,
          validation_data=(X_test,y_test),
          callbacks=[tensorboard_callback])

# preallocation of memory approach
predicted_advanced = np.empty([len(X_test),5]) # 5 refer to no. of categories
for index, test in enumerate(X_test):
    predicted_advanced[index,:] =  model.predict(np.expand_dims(test,axis=0))


# report generation
y_pred = np.argmax(predicted_advanced,axis=1)
y_true = np.argmax(y_test,axis=1)
print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true,y_pred))


#%% Save model
model.save(MODEL_SAVE_PATH)

