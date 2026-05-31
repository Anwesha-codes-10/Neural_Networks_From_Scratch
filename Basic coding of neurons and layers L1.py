inputs=[1,2,3]
weights=[0.2,0.8,-0.5]

bias=2
outputs={inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias}

print(outputs)


inputs=[1.0,2.0,3.0,2.5]
weights=[0.2,0.8,-0.5,1.0]

bias=2
outputs={inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias}
print(outputs)

inputs=[1,2,3,2.5]
weights=[[0.2,0.8,-0.5,1],
         [0.5,-0.91,0.26,-0.5],
         [-0.26,-0.27,0.17,0.07]]
biases=[2,3,4.5]

outputs={
    #Neuron 1:
    weights[0][0]*inputs[0] + weights[0][1]*inputs[1] + weights[0][2]*inputs[2] + weights[0][3]*inputs[3] + biases[0],

    #Neuron 2:
    weights[1][0]*inputs[0] + weights[1][1]*inputs[1] + weights[1][2]*inputs[2] + weights[1][3]*inputs[3] + biases[1],

    #Neuron 3:
    weights[2][0]*inputs[0] + weights[2][1]*inputs[1] + weights[2][2]*inputs[2] + weights[2][3]*inputs[3] + biases[2]
}

print(outputs)
inputs=[1,2,3,2.5]
weights=[[0.2,0.8,-0.5,1],
         [0.5,-0.91,0.26,-0.5],
         [-0.26,-0.27,0.17,0.07]]
biases=[2,3,4.5]

layer_outputs=[]

for neuron_weights, neuron_bias in zip(weights,biases):
    
    neuron_output=0

    
    for n_input, weight in zip(inputs,neuron_weights):
         neuron_output+=n_input*weight 
    neuron_output+=neuron_bias  
    layer_outputs.append(neuron_output)
print(layer_outputs)
        

