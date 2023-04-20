import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


# the calculating function
def my_function(domain):
    result = [math.sin(i/2) for i in domain]
    return result

# calculating the loss 
def calculate_loss(estimated, goal):
    return np.sum((estimated-goal)**2)**0.5/len(estimated)

# making the x points
def making_x_points(points, domain):
    x = np.linspace(domain[0], domain[1], points).reshape(-1, 1)
    return x
    
# making the y points    
def making_y_points(x):
    y = np.array(my_function(x)).reshape(-1, 1)
    return y

def train(iteration, hidden_layer, x_train, y_train):
    # doing the training here
    trained_network = MLPRegressor( hidden_layer_sizes=hidden_layer,
                                    max_iter=iteration,
                                    random_state=1,
                                    shuffle=True).fit(x_train, y_train.ravel())
    return trained_network

def diagram(x_train, y_train, x_test, y_test, y_result, loss, points):
    fig, ax = plt.subplots()
    train_plt, = plt.plot(x_train, y_train, label='Train',  linewidth=3, linestyle=':')
    test_plt,  = plt.plot(x_test, y_result, label='Test')
    expected_plt,  = plt.plot(x_test, y_test, label='Expected_result')
    ax.set_title('Mean squared loss: ' + str(round(loss, 3)))
    ax.legend(handles=[train_plt, test_plt, expected_plt])
    name = "result_" + str(points) + '.png'
    plt.savefig(name)
    plt.show()
    print()


if __name__ == "__main__":

    # making train points
    train_points = 2000
    train_domain = (0, 25)
    x_train = making_x_points(train_points, train_domain)
    y_train = making_y_points(x_train)
    
    # adding noises
    t = train_points/2
    y_train[0] = y_train[0] * 5
    y_train[train_points-1] = y_train[train_points-1] / 13
    
    # making test points
    test_points = train_points*3
    test_domain = (-5, 30)
    x_test = making_x_points(test_points, test_domain)
    y_test = making_y_points(x_test)
    
    # training 
    iterations = 2000
    hidden_layer = (20, 15, 10, 20)
    trained_network = train(iterations, hidden_layer, x_train, y_train)

    # the result of test inputs, that our network believe is true
    y_result = trained_network.predict(x_test) 

    # calculating the loss of our network result and the real result
    loss = mean_squared_error(y_result, y_test)

    # making the diagram
    diagram(x_train, y_train, x_test, y_test, y_result, loss, train_points) 