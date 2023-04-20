import sklearnProgram
from sklearn.metrics import mean_squared_error
import numpy as np
import random


# making my 2D points
def twoD_x_points(domain, points):
    x = np.array([[random.randint(domain[0], domain[1]), random.randint(domain[0], domain[1])] for _ in range(points)])
    # print(len(x))
    # print(x)
    # print(x[0])
    # print(x[0][0])
    return x
    
def twoD_y_points(x):
    y = np.array([[i[0] + i[1]] for i in x])
    return y


if __name__ == "__main__":
        
    points = 200
    domain = (0, 10)
    
    # making points 
    x_train = twoD_x_points(domain, points) 
    y_train = twoD_y_points(x_train)

    x_test = twoD_x_points(domain, points/10)
    y_test = twoD_y_points(x_test)

    # training    
    iterations = 500
    hidden_layer = (15, 15)
    trained_network = sklearnProgram.train(iterations, hidden_layer, x_train, y_train)

    # the result of test inputs, that our network believe is true
    y_result = trained_network.predict(x_test) 

    # calculating the loss of our network result and the real result
    loss = mean_squared_error(y_result, y_test)

    # making the diagram
    sklearnProgram.diagram(x_train, y_train, x_test, y_test, y_result, loss, 14) 