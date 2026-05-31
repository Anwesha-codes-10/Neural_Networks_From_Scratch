#!/usr/bin/env python
# coding: utf-8

# ### Generating non linear data

# In[17]:


from nnfs.datasets import spiral_data
import numpy as np
import nnfs
nnfs.init()
import matplotlib.pyplot as plt
X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1])
plt.show()


# In[18]:


plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()


# ### From lecture 4 I am taking the dense layer function

# In[19]:


import numpy as np
import nnfs
from nnfs.datasets import spiral_data

#Dense Layer
class Layer_Dense:
    #Layer Intialisation
    def __init__(self, n_inputs, n_neurons):
    #Intialize weights and biases
        self.weights=0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases=np.zeros((1, n_neurons))

    #Forward pass
    def forward(self,inputs):
    #Calculate output values from inputs, weights and biases
        self.output=np.dot(inputs, self.weights) + self.biases

#Create dataset
x, y = spiral_data(samples=100, classes=3)
#create dense layer with 2 input features and 3 ouyput values
dense1=Layer_Dense(2,3)
#perform a forward pass of our training data through this layer
dense1.forward(x)
print(dense1.output[:5])
    
    
    


# ### Activation Functions: ReLU

# In[20]:


import numpy as np
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = np.maximum(0, inputs) #here i have created the relu function
print(output)


# ### Making a class for the relu funcion

# In[21]:


#Now implementing the relu function
class Activation_ReLU:
    #Forward pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) #calculating output values from the input


# ### Softmax function implementation

# In[22]:


inputs = [[1, 2, 3, 2.5],
          [2., 5., -1., 2],
          [-1.5, 2.7, 3.3, -0.8]]

#get unnormalized probabilities
exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
#normalizing each sample
probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(probabilities)
np.sum(probabilities, axis=1)


# ### Now implementing the above function in a class

# In[23]:


#Softmax activation
class Activation_Softmax:
    #forward pass
    def forward(self, inputs):
        #get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #normalizing each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
                            


# ### Note: for all the proper mathematics behind this refer the lecture 6 notes kindly 

# In[26]:


#Create dataset
X, y = spiral_data(samples=100, classes=3)

#create dense layer with 2 inputs and 3 ouput values
dense1 = Layer_Dense(2, 3)

#create a relu activation (to be used in dense layer)
activation1 = Activation_ReLU()

#create a second dense layer with 3 input features (as we have taken output of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

#Create Softmax activation (to be used with the previous dense layer):
activation2 = Activation_Softmax()

#Make a forward pass of our training data through this layer
dense1.forward(X)

#Make a forward pass through activation function
#takes the output the first dense layer
activation1.forward(dense1.output)

#Make a forward pass through the second dense layer
#it takes outputs of first activation layer as inputs
dense2.forward(activation1.output)

#make forward pass through activation function
#it takes the output of second dense layer here
activation2.forward(dense2.output)


#final output but for few samples
print(activation2.output[:5])


# In[ ]:




