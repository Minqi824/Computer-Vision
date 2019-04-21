# =============================================================================
# #a simple RNN
# =============================================================================
import numpy as np

#timesteps
timesteps=100
#input's dimension
input_features=32
#output's dimension
output_features=64

#input
inputs=np.random.random((timesteps,input_features))

type(inputs)
#100*32
inputs.shape

#initial state (same length as output)
state_t=np.zeros((output_features,))

#weight matrix
W=np.random.random((output_features,input_features))
U=np.random.random((output_features,output_features))

b=np.random.random((output_features,))

sucessive_output=[]
for input_t in inputs:
    #here we use a tanh activation function
    output_t=np.tanh(np.dot(W,input_t)+np.dot(U,state_t)+b)
    #store the output result
    sucessive_output.append(output_t)
    #recursively update the state
    state_t=output_t
    
#transform the list to a matrix(two dimension array)
final_output_sequence=np.stack(sucessive_output,axis=0)
final_output_sequence.shape    
final_output_sequence
    
# =============================================================================
# #a simple RNN layer from keras
# =============================================================================
#from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

model=Sequential()
model.add(Embedding(10000,32))
#return the last sequence
model.add(SimpleRNN(32))
model.summary()

model=Sequential()
model.add(Embedding(10000,32))
#return the entire sequence
model.add(SimpleRNN(32,return_sequences=True))
model.summary()

#stack simpleRNN
model=Sequential()
model.add(Embedding(10000,32))
model.add(SimpleRNN(32,return_sequences=True))
model.add(SimpleRNN(32,return_sequences=True))
model.add(SimpleRNN(32,return_sequences=True))
#the last layer only return the last sequence
model.add(SimpleRNN(32))
model.summary()

# =============================================================================
# #use RNN for classification (IMDB dataset)
# =============================================================================
from keras.datasets import imdb
from keras.preprocessing import sequence    

max_features=10000
maxlen=500
batch_size=32

(input_train,y_train),(input_test,y_test)=imdb.load_data(num_words=max_features)

print(len(input_train),'train sequences')
print(len(input_test),'test sequences')

print('Pad sequences (samples x time)')
input_train=sequence.pad_sequences(input_train,maxlen=maxlen)
input_test=sequence.pad_sequences(input_test,maxlen=maxlen)

print('input_train shape:',input_train.shape)
print('input_test shape:',input_test.shape)

#train the model
from keras.layers import Dense

model=Sequential()
model.add(Embedding(max_features,32))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))#classification problem

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
#record the training process
history=model.fit(input_train,y_train,epochs=10,batch_size=128,validation_split=0.2)

#loss value in training and validation process
import matplotlib.pyplot as plt

#training accuracy
acc=history.history['acc']
#validation accuracy
val_acc=history.history['val_acc']
#training loss
loss=history.history['loss']
#validation loss
val_loss=history.history['val_loss']

#iteration
epochs=range(1,len(acc)+1)
#accuracy value in training and validation process
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

#loss value in training and validation process
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
#plt.figure()

# =============================================================================
# #long short-term memory
# =============================================================================
from keras.layers import LSTM

model=Sequential()
model.add(Embedding(max_features,32))
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))

#compile the model
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

history=model.fit(input_train,y_train,epochs=10,batch_size=128,validation_split=0.2)

#training accuracy
acc=history.history['acc']
#validation accuracy
val_acc=history.history['val_acc']
#training loss
loss=history.history['loss']
#validation loss
val_loss=history.history['val_loss']

#iteration
epochs=range(1,len(acc)+1)
#accuracy value in training and validation process
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

#loss value in training and validation process
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
