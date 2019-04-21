# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:34:15 2019

@author: Administrator
"""
# =============================================================================
# a simple neural network model
# =============================================================================
#import IMDB dataset
from keras.datasets import imdb

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

#training data and label
train_data[0]
train_labels[0]

#the dataset only include the most 10000 frequent words
max([max(sequence) for sequence in train_data])

#the words and it's corresponding index
word_index=imdb.get_word_index()

reverse_word_index=dict([(value,key) for (key,value) in word_index.items()])

#decode the review(value to words)
decoded_review=''.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])

#one-hot encode
import numpy as np

#dimension=10000 is the default value
def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    #two folds list
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1
    return results

#vectorize the training set(X)
x_train=vectorize_sequences(train_data)
#vectorize the test set(X)
x_test=vectorize_sequences(test_data)

x_train[0]

#vecotrize the labels
y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')

#define model's structure:1 hidden layer which include 16 units
from keras import models
from keras import layers

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#test the model
x_val=x_train[:10000]
partial_x_train=x_train[10000:]

y_val=y_train[:10000]
partial_y_train=y_train[10000:]

history=model.fit(partial_x_train,
                  partial_y_train,
                  epochs=20,
                  batch_size=512,
                  validation_data=(x_val,y_val))



#plot the loss in training process and validating process
import matplotlib.pyplot as plt

history_dict=history.history
history_dict.keys()

loss_values=history_dict['loss']
val_loss_values=history_dict['val_loss']

epochs=range(1,len(loss_values)+1)

plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#Training and Validation accuracy
#clear the current picture

plt.clf()

acc=history_dict['acc']
val_acc=history_dict['val_acc']

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#use 4 epochs to train a new model
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train,
                  partial_y_train,
                  epochs=4,
                  batch_size=512,
                  validation_data=(x_val,y_val))

#test the new model on test data
print(model.evaluate(x_test,y_test))

#get the predictive probabilities on test set
model.predict(x_test)

# =============================================================================
# 
# =============================================================================
#original model

from keras import models
from keras import layers

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#smaller model
model=models.Sequential()
model.add(layers.Dense(4,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(4,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#bigger model
model=models.Sequential()
model.add(layers.Dense(512,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))


#regularization
from keras import regularizers

model=models.Sequential()
#L1 regularization
model.add(layers.Dense(16,kernel_regularizer=regularizers.l1(0.001),
                       activation='relu',input_shape=(10000,)))
#dropout regularization
model.add(layers.Dropout(0.5))#dropout rate is 0.5
#L2 regularization
model.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.001),
                       activation='relu',input_shape=(10000,)))
#L1 + L2 regularization
model.add(layers.Dense(16,kernel_regularizer=regularizers.l1_l2(l1=0.001,l2=0.001),
                       activation='relu',input_shape=(10000,)))
model.add(layers.Dense(1,activation='sigmoid'))