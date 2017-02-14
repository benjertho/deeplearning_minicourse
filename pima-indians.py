# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:56:00 2017

@author: thompson
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

seed=7
np.random.seed(seed)

dataset = np.loadtxt('data/pima_indians/pima-indians-diabetes.data', delimiter=',')
split = int(0.9 * len(dataset))

X_train=dataset[:split,:8]
Y_train=dataset[:split,8]
X_test=dataset[split:,:8]
Y_test=dataset[split:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train,Y_train,nb_epoch=150,batch_size=10)

scores=model.evaluate(X_test,Y_test)
print ("\n\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

