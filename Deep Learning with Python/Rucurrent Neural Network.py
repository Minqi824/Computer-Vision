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

# =============================================================================
# jena climate forecasting
# =============================================================================
import os
fname=os.path.join('E:/Program/Python/Project_Improved Stacking Framework/Code/Deep Learning with Python','jena_climate_2009_2016.csv')

f=open(fname)
data=f.read()
f.close()

lines=data.split('\n')
header=lines[0].split(',')
lines=lines[1:]

print(header)
print(len(lines))

#transform the data to array
import numpy as np

float_data=np.zeros((len(lines),len(header)-1))
float_data.shape

for i,line in enumerate(lines):
    values=[float(x) for x in line.split(',')[1:]]
    float_data[i,:]=values

#the temp series
from matplotlib import pyplot as plt
temp=float_data[:,1]

plt.plot(range(len(temp)),temp)

#the first 10 days
plt.plot(range(1440),temp[:1440])
    
#data preprocessing
mean=float_data[:200000].mean(axis=0)
float_data-=mean

std=float_data[:20000].std(axis=0)
float_data/=std

#data generator
def generator(data,lookback,delay,min_index,max_index,shuffle=False,batch_size=128,step=6):
    if max_index is None:
        max_index=len(data)-delay-1
    i=min_index+lookback
    while 1:
        if shuffle:
            rows=np.random.randint(min_index+lookback,max_index,size=batch_size)
        else:
            if i+batch_size>=max_index:
                i=min_index+lookback
            rows=np.arange(i,min(i+batch_size,max_index))
            i+=len(rows)
        
        samples=np.zeros((len(rows),lookback//step,data.shape[-1]))
        targets=np.zeros((len(rows),))
        for j,row in enumerate((len(rows),)):
            indices=range(rows[j]-lookback,rows[j],step)
            samples[j]=data[indices]
            targets[j]=data[rows[j]+delay][1]
        yield samples,targets
        
lookback=1440
step=6
delay=144
batch_size=128

train_gen=generator(float_data,lookback=lookback,delay=delay,min_index=0,max_index=200000,shuffle=True,step=step,batch_size=batch_size)
val_gen=generator(float_data,lookback=lookback,delay=delay,min_index=200001,max_index=300000,step=step,batch_size=batch_size)
test_gen=generator(float_data,lookback=lookback,delay=delay,min_index=300001,max_index=None,step=step,batch_size=batch_size)

val_steps=(300000-200001-lookback)//batch_size
test_steps=(len(float_data)-300001-lookback)//batch_size

# =============================================================================
# #benchmark
# =============================================================================
def evaluate_naive_method():
    batch_maes=[]
    for step in range(val_steps):
        samples,targets=next(val_gen)
        preds=samples[:,-1,1]
        mae=np.mean(np.abs(preds-targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))

#benchmark mae
evaluate_naive_method()

# =============================================================================
# a simple neural network
# =============================================================================
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model=Sequential()
model.add(layers.Flatten(input_shape=(lookback//step,float_data.shape[-1])))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(),loss='mae')
history=model.fit_generator(train_gen,steps_per_epoch=500,epochs=20,validation_data=val_gen,validation_steps=val_steps)

#visulize the result
import matplotlib.pyplot as plt

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(loss)+1)

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend

plt.show()

# =============================================================================
# GRU-RNN
# =============================================================================
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model=Sequential()
model.add(layers.GRU(32,input_shape=(None,float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(),loss='mae')

history=model.fit_generator(train_gen,steps_per_epoch=500,epochs=20,validation_data=val_gen,validation_steps=val_steps)

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(loss)+1)

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend

plt.show()

# =============================================================================
# RNN-GRU with dropout
# =============================================================================
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model=Sequential()
model.add(layers.GRU(32,dropout=0.2,recurrent_dropout=0.2,input_shape=(None,float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(),loss='mae')

history=model.fit_generator(train_gen,steps_per_epoch=500,epochs=40,validation_data=val_gen,validation_steps=val_steps)

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(loss)+1)

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend

plt.show()

# =============================================================================
# stack RNN-GRU with dropout
# =============================================================================
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model=Sequential()
model.add(layers.GRU(32,dropout=0.1,recurrent_dropout=0.5,return_sequences=True,input_shape=(None,float_data.shape[-1])))
model.add(layers.GRU(64,activation='relu',dropout=0.1,recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(),loss='mae')

history=model.fit_generator(train_gen,steps_per_epoch=500,epochs=40,validation_data=val_gen,validation_steps=val_steps)

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(loss)+1)

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend

plt.show()

# =============================================================================
# bidirectional RNN-LSTM
# =============================================================================
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model=Sequential()
model.add(layers.Bidirectional(layers.LSTM(32),input_shape(None,float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(),loss='mae')

history=model.fit_generator(train_gen,steps_per_epoch=500,epochs=40,validation_data=val_gen,validation_steps=val_steps)

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(loss)+1)

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend

plt.show()