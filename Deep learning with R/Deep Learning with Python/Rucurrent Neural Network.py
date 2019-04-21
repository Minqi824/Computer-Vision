#a simple RNN
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
    
    
    
    