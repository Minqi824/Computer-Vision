# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 11:13:26 2019

@author: lenovo
"""

#Deep Learning project from https://www.kaggle.com/jmq19950824/prediction-stock-index-deep-learning
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt

#data resource:kaggle kernel platform
data=pd.read_csv('../input/stock-index-prediction-both-labels-and-features/SP500.csv')
#delete the Date variable
data=data.iloc[:,1:]
pred_results=pd.DataFrame()

X_train=data.iloc[:,1:].iloc[0:1958,:]
Y_train=data.iloc[:,0:1].iloc[0:1958,:]
X_test=data.iloc[:,1:].iloc[1958:,:]
Y_test=data.iloc[:,0:1].iloc[1958:,:]
#preprocess the data
scaler=MinMaxScaler()
scaler1=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_train1=scaler1.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
X_test1=scaler1.fit_transform(X_test)

timestep=20
#reshape the data for 3d
X_deep_matrix=np.append(X_train,X_test,axis=0)
#transform the matrix data to tensor data
X_deep_tensor=np.empty((X_deep_matrix.shape[0]-timestep+1,timestep,X_deep_matrix.shape[1]))
for i  in range(timestep-1,X_deep_matrix.shape[0]):
    X_deep_tensor[i-timestep+1]=X_deep_matrix[i-timestep+1:i+1,:]
del X_deep_matrix

X_train_deep=X_deep_tensor[:X_deep_tensor.shape[0]-Y_test.shape[0]]
X_test_deep=X_deep_tensor[X_deep_tensor.shape[0]-Y_test.shape[0]:]

del X_deep_tensor

Y_train_deep=Y_train[timestep-1:len(Y_train)]

Model_Deep_DNN=Sequential()
Model_Deep_DNN.add(layers.Dense(32,input_shape=(X_train.shape[1],)))
Model_Deep_DNN.add(layers.Dropout(0.1))
Model_Deep_DNN.add(layers.Dense(32))
Model_Deep_DNN.add(layers.Dropout(0.1))
Model_Deep_DNN.add(layers.Dense(32))
Model_Deep_DNN.add(layers.Dropout(0.1))
Model_Deep_DNN.add(layers.Dense(1,activation='sigmoid'))

Model_Deep_DNN.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['accuracy'])
history=Model_Deep_DNN.fit(X_train,Y_train,batch_size=128,epochs=100,validation_split=0.2)

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'ro',label='Validation acc')
plt.plot(epochs,loss,'b',label='Training loss')
plt.plot(epochs,val_loss,'r',label='Validation loss')
plt.legend() 
plt.figure()

#compare two different standardization 
X_deep_matrix1=np.append(X_train1,X_test1,axis=0)
#transform the matrix data to tensor data
X_deep_tensor1=np.empty((X_deep_matrix1.shape[0]-timestep+1,timestep,X_deep_matrix1.shape[1]))
for i  in range(timestep-1,X_deep_matrix1.shape[0]):
    X_deep_tensor1[i-timestep+1]=X_deep_matrix1[i-timestep+1:i+1,:]
del X_deep_matrix1

X_train_deep1=X_deep_tensor1[:X_deep_tensor1.shape[0]-Y_test.shape[0]]
X_test_deep1=X_deep_tensor1[X_deep_tensor1.shape[0]-Y_test.shape[0]:]

del X_deep_tensor1

Y_train_deep1=Y_train[timestep-1:len(Y_train)]

Model_Deep_DNN1=Sequential()
Model_Deep_DNN1.add(layers.Dense(32,input_shape=(X_train1.shape[1],)))
Model_Deep_DNN1.add(layers.Dropout(0.1))
Model_Deep_DNN1.add(layers.Dense(32))
Model_Deep_DNN1.add(layers.Dropout(0.1))
Model_Deep_DNN1.add(layers.Dense(32))
Model_Deep_DNN1.add(layers.Dropout(0.1))
Model_Deep_DNN1.add(layers.Dense(1,activation='sigmoid'))

Model_Deep_DNN1.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['accuracy'])
history1=Model_Deep_DNN1.fit(X_train1,Y_train,batch_size=128,epochs=100,validation_split=0.2)

acc1=history1.history['acc']
val1_acc=history1.history['val_acc']
loss1=history1.history['loss']
val1_loss=history1.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc1,'bo',label='Training acc')
plt.plot(epochs,val1_acc,'ro',label='Validation acc')
plt.plot(epochs,loss1,'b',label='Training loss')
plt.plot(epochs,val1_loss,'r',label='Validation loss')
plt.legend() 
plt.show()

'''
----------------------------------------------------------------------------------------------------------------------------------------
#The MinMaxScaler way's test accuracy is more stable and there is no outliers in this dataset,so here we think the MinMax transform is more suitble. 
----------------------------------------------------------------------------------------------------------------------------------------
'''

timestep=20
#reshape the data for 3d
X_deep_matrix=np.append(X_train,X_test,axis=0)
#transform the matrix data to tensor data
X_deep_tensor=np.empty((X_deep_matrix.shape[0]-timestep+1,timestep,X_deep_matrix.shape[1]))
for i  in range(timestep-1,X_deep_matrix.shape[0]):
    X_deep_tensor[i-timestep+1]=X_deep_matrix[i-timestep+1:i+1,:]
del X_deep_matrix

X_train_deep=X_deep_tensor[:X_deep_tensor.shape[0]-Y_test.shape[0]]
X_test_deep=X_deep_tensor[X_deep_tensor.shape[0]-Y_test.shape[0]:]

del X_deep_tensor

Y_train_deep=Y_train[timestep-1:len(Y_train)]


#Recursive Neural Network(RNN)
Model_Deep_RNN=Sequential()
Model_Deep_RNN.add(layers.SimpleRNN(32,dropout=0.1,recurrent_dropout=0.1,return_sequences=True,
                                    input_shape=(X_train_deep.shape[1],X_train_deep.shape[2])))
Model_Deep_RNN.add(layers.SimpleRNN(32,dropout=0.1,recurrent_dropout=0.1,return_sequences=True))
Model_Deep_RNN.add(layers.SimpleRNN(32,dropout=0.1,recurrent_dropout=0.1,return_sequences=False))
Model_Deep_RNN.add(layers.Dense(1,activation='sigmoid'))

Model_Deep_RNN.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['accuracy'])
history=Model_Deep_RNN.fit(X_train_deep,Y_train_deep,batch_size=128,epochs=100,validation_split=0.2)

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'ro',label='Validation acc')
plt.plot(epochs,loss,'b',label='Training loss')
plt.plot(epochs,val_loss,'r',label='Validation loss')
plt.legend() 
plt.show()



#bidirectional RNN (BRNN)
timestep=20
#reshape the data for 3d
X_deep_matrix=np.append(X_train,X_test,axis=0)
#transform the matrix data to tensor data
X_deep_tensor=np.empty((X_deep_matrix.shape[0]-timestep+1,timestep,X_deep_matrix.shape[1]))
for i  in range(timestep-1,X_deep_matrix.shape[0]):
    X_deep_tensor[i-timestep+1]=X_deep_matrix[i-timestep+1:i+1,:]
del X_deep_matrix

X_train_deep=X_deep_tensor[:X_deep_tensor.shape[0]-Y_test.shape[0]]
X_test_deep=X_deep_tensor[X_deep_tensor.shape[0]-Y_test.shape[0]:]

del X_deep_tensor

Y_train_deep=Y_train[timestep-1:len(Y_train)]

Model_Deep_BRNN=Sequential()
Model_Deep_BRNN.add(layers.Bidirectional(layers.SimpleRNN(5,dropout=0.1,recurrent_dropout=0.1,return_sequences=True,
                                     input_shape=(X_train_deep.shape[1],X_train_deep.shape[2]))))
Model_Deep_BRNN.add(layers.Bidirectional(layers.SimpleRNN(5,dropout=0.1,recurrent_dropout=0.1,return_sequences=True)))
Model_Deep_BRNN.add(layers.Bidirectional(layers.SimpleRNN(5,dropout=0.1,recurrent_dropout=0.1,return_sequences=False)))
Model_Deep_BRNN.add(layers.Dense(1,activation='sigmoid'))

Model_Deep_BRNN.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['accuracy'])
history=Model_Deep_BRNN.fit(X_train_deep,np.array(Y_train_deep),batch_size=128,epochs=100,validation_split=0.2)

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'ro',label='Validation acc')
plt.plot(epochs,loss,'b',label='Training loss')
plt.plot(epochs,val_loss,'r',label='Validation loss')
plt.legend() 
plt.show()




#RNN-LSTM
Model_Deep_RNNLSTM=Sequential()
Model_Deep_RNNLSTM.add(layers.LSTM(5,dropout=0.5,recurrent_dropout=0.5,return_sequences=True,
                                    input_shape=(X_train_deep.shape[1],X_train_deep.shape[2])))
Model_Deep_RNNLSTM.add(layers.LSTM(5,dropout=0.5,recurrent_dropout=0.5,return_sequences=True))
Model_Deep_RNNLSTM.add(layers.LSTM(5,dropout=0.5,recurrent_dropout=0.5,return_sequences=False))
Model_Deep_RNNLSTM.add(layers.Dense(1,activation='sigmoid'))

Model_Deep_RNNLSTM.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['accuracy'])
history=Model_Deep_RNNLSTM.fit(X_train_deep,Y_train_deep,batch_size=128,epochs=20,validation_split=0.2)

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'ro',label='Validation acc')
plt.plot(epochs,loss,'b',label='Training loss')
plt.plot(epochs,val_loss,'r',label='Validation loss')
plt.legend() 
plt.show()




#RNN-GRU
Model_Deep_RNNGRU=Sequential()
Model_Deep_RNNGRU.add(layers.GRU(3,dropout=0.5,recurrent_dropout=0.5,return_sequences=True,
                                    input_shape=(X_train_deep.shape[1],X_train_deep.shape[2])))
Model_Deep_RNNGRU.add(layers.GRU(3,dropout=0.5,recurrent_dropout=0.5,return_sequences=True))
Model_Deep_RNNGRU.add(layers.GRU(3,dropout=0.5,recurrent_dropout=0.5,return_sequences=False))
Model_Deep_RNNGRU.add(layers.Dense(1,activation='sigmoid'))

Model_Deep_RNNGRU.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['accuracy'])
history=Model_Deep_RNNGRU.fit(X_train_deep,Y_train_deep,batch_size=128,epochs=20,validation_split=0.2)

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'ro',label='Validation acc')
plt.plot(epochs,loss,'b',label='Training loss')
plt.plot(epochs,val_loss,'r',label='Validation loss')
plt.legend() 
plt.show()

#print the predict results of four DL Algorithms
pred_Deep=pd.concat([pd.DataFrame(Model_Deep_RNN.predict(X_test_deep)),
                     pd.DataFrame(Model_Deep_BRNN.predict(X_test_deep)),
                     pd.DataFrame(Model_Deep_RNNLSTM.predict(X_test_deep)),
                     pd.DataFrame(Model_Deep_RNNGRU.predict(X_test_deep))],axis=1)

pred_Deep[pred_Deep>0.5]=1
pred_Deep[pred_Deep<=0.5]=0

pred_result=pd.concat([pd.DataFrame(Y_test),pred_Deep],axis=1)

#rename
pred_result.columns=['LABEL','RNN_Pred','BRNN_Pred','RNNLSTM_Pred','RNNGRU_Pred']
pred_result