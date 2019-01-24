"""
Created on Tue Jan  22 11:13:36 2019

@author: akash
"""


from __future__ import print_function, division
from builtins import range, input

from keras.models import Model
from keras.layers import Input, LSTM, GRU
import numpy as np 
import matplotlib.pyplot as plt 

sequence_length = 8 # sequence length
input_dimentionality = 2 # Input dimentionality
hidden_layer = 3 # Hidden layer

X = np.random.randn(1, sequence_length, input_dimentionality) # dummy input , 1 sample of size sequence_length * input_dimentionality

def lstm1():
    input_ = Input(shape=(sequence_length, input_dimentionality)) # input layer size = sequence_length * input_dimentionality
    rnn = LSTM(hidden_layer, return_state=True)
    x = rnn(input_)
    # an LSTM has two state H and C
    model = Model(inputs=input_, outputs=x)
    o, h , c = model.predict(X) # o = actual output, h = hidden state, c = cell state
    print('o: ', o)
    print('h: ', h)
    print('c: ', c)

def lstm2():
    input_ = Input(shape=(sequence_length, input_dimentionality))
    rnn = LSTM(hidden_layer, return_state=True, return_sequences=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h, c = model.predict(X)
    print('o: ',o)
    print('h: ',h)
    print('c: ',c)


def gru1():
    input_ = Input(shape=(sequence_length, input_dimentionality))
    rnn = GRU(hidden_layer, return_state = True)
    x = rnn(input_)

    model = Model(inputs = input_, outputs = x)
    o, h = model.predict(X) # o = actual output, h = hidden state
    print('o: ', o)
    print('h: ', h)

def gru2():
    input_ = Input(shape=(sequence_length, input_dimentionality))
    rnn = GRU(hidden_layer, return_state = True, return_sequences=True)
    x = rnn(input_)

    model = Model(inputs = input_, outputs = x)
    o, h = model.predict(X)
    print('o: ', o)
    print('h: ', h)


print('lstm1')
lstm1()
print('lstm2')
lstm2()
print('gru1')
gru1()
print('gru2')
gru2()

