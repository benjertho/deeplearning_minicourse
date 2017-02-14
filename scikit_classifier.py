# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:21:26 2017

@author: thompson
"""

from keras.models import Sequential
from keras.layers import Dense


def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
model = KerasClassifier