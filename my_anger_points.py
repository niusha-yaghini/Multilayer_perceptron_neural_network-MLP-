import sklearnProgram
from sklearn.metrics import mean_squared_error
import numpy as np


# making my anger points
def anger_x_train_points():
    x = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]).reshape(-1, 1)
    return x
    
def anger_y_train_points():
    y = np.array([6, 4.7, 3.6, 2.9, 3.1, 4.8, 0.4, 0.9, 1.1, 4.4, 4.3, 6.1, 6.9, 7.4]).reshape(-1, 1)
    return y

def anger_x_test_points():
    x = np.array([3, 9, 11, 15, 21]).reshape(-1, 1)
    return x
    
def anger_y_test_points():
    y = np.array([4, 3.5, 2.5, 3.6, 5.3]).reshape(-1, 1)
    return y


if __name__ == "__main__":
        
    # making points 
    x_train = anger_x_train_points() #14
    y_train = anger_y_train_points()

    x_test = anger_x_test_points() #5
    y_test = anger_y_test_points()

    # training    
    iterations = 1000
    hidden_layer = (10, 10)
    trained_network = sklearnProgram.train(iterations, hidden_layer, x_train, y_train)

    # the result of test inputs, that our network believe is true
    y_result = trained_network.predict(x_test) 

    # calculating the loss of our network result and the real result
    loss = mean_squared_error(y_result, y_test)

    # making the diagram
    sklearnProgram.diagram(x_train, y_train, x_test, y_test, y_result, loss, 14) 
