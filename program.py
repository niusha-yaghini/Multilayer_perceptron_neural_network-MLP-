import numpy as np
from random import random


class MLP(object):
#a multilayer perception class

    def __init__(self, num_inputs=2, hidden_layers=[3, 3], num_outputs=1):
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        
        #create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]
        
        #create random connection weights for the layers(matrix)
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights
        
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations
        
        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros(layers[i], layers[i+1])
            derivatives.append(d)
        self.derivatives = derivatives
        
        
    def forward_propagates(self, inputs):
        #computes forward propagation of the network based on input
        #Return: (NDarray) activations
        
        
        #the input layer activation is just the input itself
        activations = inputs
        self.activations[0] = inputs
        
        #iterate through the network layers
        for i, w in enumerate(self.weights):
            #calculate matrix multiplication between previos activation and weight matrix
            net_inputs = np.dot(activations, w)
            
            #apply sigmoid activation function
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations
            
        #return output layer activation
        return activations
        
        