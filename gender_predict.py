# -*- coding: utf-8 -*-
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# load tokenizer
nwords=40
tokenizer = Tokenizer(num_words=nwords)
with open('lib/tokenizer_letter.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle) 

# load model
json_file = open('lib/model_gender_letter.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# load weights
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('lib/weights_gender_letter_cnn-lstm2.hdf5')


# preprocess
def preprocess(nama):
    # nama lowercase
    nama=nama.lower()
    
    # nama di split jadi per huruf
    nama= list(nama)
    
    # nama split di input ke list, karena input fungsi pad_sequences() harus berbentuk list
    tmp=[]
    tmp.append(nama)
    tmp.append(nama)
    
    # tokenizing
    sequences = tokenizer.texts_to_sequences(tmp) 
    
    #padding
    input_seq = pad_sequences(sequences, maxlen=40)
    return input_seq

global graph
graph = tf.get_default_graph()

# predict
def predict(text):
    input=text
    print(input)
    
    # preprocess
    input=preprocess(input)

    # predict classes
    with graph.as_default():
        prediction = loaded_model.predict_classes(input).tolist()
    
    return json.dumps(prediction[0])

#nama = 'Aditya Rizky'
#print(predict(nama))