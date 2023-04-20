import numpy as np
from random import *

class MLP(object):
    #A Multilayer Perceptron class.

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
        #Constructor for the MLP

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        # save derivatives per layer
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations


    def forward_propagate(self, inputs):
        #Computes forward propagation of the network based on input signals.

        # the input layer activation is just the input itself
        activations = inputs

        # save the activations for backpropogation
        self.activations[0] = activations

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

            # save the activations for backpropogation
            self.activations[i + 1] = activations

        # return output layer activation
        return activations


    def back_propagate(self, loss, verbose=False):
        #Backpropogates an loss signal.
        # Returns: loss (ndarray): The final loss of the input

        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):

            # get activation for previous layer
            activations = self.activations[i+1]

            # apply sigmoid derivative function
            delta = loss * self._sigmoid_derivative(activations)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(current_activations.shape[0],-1)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # backpropogate the next loss
            loss = np.dot(delta, self.weights[i].T)
            
        
            if verbose:
                print("Derivatives for w{}: {}".format(i, self.derivatives[i]))


    def train(self, inputs, targets, iteration, learning_rate, verbose=False):
        #Trains model running forward prop and backprop

        # now enter the training loop
        for i in range(iteration):
            sum_loss = 0

            # iterate through all the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # activate the network!
                output = self.forward_propagate(input)

                loss = target - output

                self.back_propagate(loss)

                # now perform gradient descent on the derivatives
                # (this will update the weights
                self.gradient_descent(learning_rate)

                # keep track of the MSE for reporting later
                sum_loss += self._mse(target, output)

            if verbose:
                # Iteration complete, report the training loss
                print("Loss: {} at iteration {}".format(sum_loss / len(items), i+1))

        print("Training complete!")
        print("=====")


    def gradient_descent(self, learningRate=1):
        #Learns by descending the gradient

        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate


    def _sigmoid(self, x):
        #Sigmoid activation function
        y = 1.0 / (1 + np.exp(-x))
        return y


    def _sigmoid_derivative(self, x):
        #Sigmoid derivative function
        return x * (1.0 - x)


    def _mse(self, target, output):
        #Mean Squared loss function
        return np.average((target - output) ** 2)


if __name__ == "__main__":

    # create a dataset to train a network for the sum operation

    # random.randint --> returns int
    # random.uniform --> return floats

    train_items = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    train_targets = np.array([[i[0] + i[1]] for i in train_items])

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(2, [5], 1)

    # train network
    mlp.train(train_items, train_targets, 50, 0.1)

    # create dummy data
    test_input = np.array([0.1, 0.4])
    # test_target = np.array([5.5])

    # get a prediction
    output = mlp.forward_propagate(test_input)

    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))
    # print(f"Our network believes that {input[0]} + {input[1]} is equal to {output[0]}")

    print()