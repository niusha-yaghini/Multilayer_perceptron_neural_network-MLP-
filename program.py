import numpy as np
from random import random

#a multilayer perception class
class MLP(object):
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
        
        
    