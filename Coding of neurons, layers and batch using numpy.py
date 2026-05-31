import numpy as np

inputs=[1,2,3]
weights=[0.2,0.8,-0.5]

bias=2
outputs=np.dot(weights,inputs) + bias


print(outputs)

inputs=[1.0,2.0,3.0,2.5]
weights=[0.2,0.8,-0.5,1.0]

bias=2
outputs=np.dot(weights, inputs) + bias
print(outputs)

inputs=[1,2,3,2.5]
weights=[[0.2,0.8,-0.5,1],
         [0.5,-0.91,0.26,-0.5],
         [-0.26,-0.27,0.17,0.07]]
biases=[2,3,4.5]

layer_outputs = np.dot(weights, inputs) + biases
print(layer_outputs)

inputs=[[1,2,3,2.5],
        [2.,5.,-1.,2],
        [-1.5,2.7,3.3,-0.8]]
weights=[[0.2,0.8,-0.5,1],
         [0.5,-0.91,0.26,-0.5],
         [-0.26,-0.27,0.17,0.07]]
biases=[2,3,4.5]

outputs = np.dot(inputs, np.array(weights).T) + biases
print(outputs)

import numpy as np
inputs=[[1,2,3,2.5],
        [2.,5.,-1.,2],
        [-1.5,2.7,3.3,-0.8]]
weights1=[[0.2,0.8,-0.5,1],
         [0.5,-0.91,0.26,-0.5],
         [-0.26,-0.27,0.17,0.07]]
biases1=[2,3,4.5]

weights2=[[0.1, -0.14, 0.5],
         [-0.5, 0.12, -0.33],
         [-0.44, 0.73, -0.13]]
biases2=[-1,-2,0.5]

i=np.array(inputs)
w1=np.array(weights1)
b1=np.array(biases1)
w2=np.array(weights2)
b2=np.array(biases2)

layer1_outputs=np.dot(i,w1.T) + b1
layer2_outputs=np.dot(layer1_outputs,w2.T) + b2

print(layer2_outputs)
