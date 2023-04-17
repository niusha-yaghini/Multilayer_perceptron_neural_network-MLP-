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
        
    
    def back_propagate(self, loss, verbose=False):
        
        #dE/dw_i = (y - a_(i+1)) s'(h_(i+1)) a_i
        #s'(h_(i+1)) = s(h_(i+1))(1 - s(h_(i+1)))
        #s(h_(i+1)) = a_(i+1)
        
        #dE/dw_(i+1) = (y - a_(i+1)) s'(h_(i+1)) w_i s'(h_i) a_(i-1)
        
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = loss * self._sigmoid_derivaties(activations) #NDarray((0.1, 0.2)) --> NDarray(((0.1, 0.2)))
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]  #NDarray((0.1, 0.2)) --> NDarray(((0.1), (0.2)))
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            loss = np.dot(delta, self.weights[i].T)
            
            if verbose:
                print("Derivatives for w{}: {}".format(i, self.derivatives[i]))
                
        return loss
    
    
    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate
            
            
    def train(self, inputs, targets, iterations, learning_rate, verbose=False):
        
        for i in range(iterations):
            sum_loss = 0
            for input, target in zip(inputs, targets):
                
                #forward propagation
                output = self.forward_propagates(input)
                
                #calculate loss
                loss = target - output
                
                #back propagation
                self.back_propagate(loss)
                
                #apply gradient descent
                self.gradient_descent(learning_rate)
                
                sum_loss += self._asl(target, output)
                
                
            #print loss each iteration
            if(verbose):
                print("Loss: {} at iteration {}".format(sum_loss/len(inputs), i))
    
                
    #average squad loss 
    def _asl(self, target, output):
        return np.average((target-output)**2)
    
    
    def _sigmoid_derivative(self, x):
        return x * (0.1 - x)
            
            
    def _sigmoid(self, x):
        #sigmoid activation function
        
        y = 1.0 / (1 + np.exp(-x))   
        return y
    
    
         
        