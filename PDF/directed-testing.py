from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

import pandas as pd
import sys


covfile = sys.argv[1]
trainfile = sys.argv[2]
covdata = numpy.loadtxt(covfile,  delimiter=",")


df = pd.read_csv(trainfile)
df = df.iloc[:, 2:]
outputdata = df.as_matrix(df)
# training data is the output
output_dim = numpy.shape(outputdata)[1]

input_dim = numpy.shape(covdata)[1]
print input_dim
# create model
model = Sequential()
model.add(Dense(input_dim, input_dim=input_dim, activation='relu'))
model.add(Dense(input_dim*2, activation='tanh'))
model.add(Dense(output_dim, activation='relu'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(covdata, outputdata, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
