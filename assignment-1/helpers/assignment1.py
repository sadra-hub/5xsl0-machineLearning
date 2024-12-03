import numpy as np

def ex11_generate_linearsystem(M:int, N:int):

    # generate X
    X = np.random.rand(N,M)

    # generate theta
    theta = np.random.rand(M)

    # calculate y
    y = X @ theta

    # return matrices
    return X, y, theta

def ex121_generate_data():

    # specify \theta 
    theta = 0.25

    # generate some input data
    x = np.random.uniform(-5, 5, 50)

    # calculate output
    y = x*theta + 0.5 + 0.2*np.random.randn(50)

    # return data
    return np.expand_dims(x, axis=1), y

def ex123_generate_data():

    # specify \theta 
    theta = [0.25, -0.1]

    # generate some input data
    x = np.random.uniform(-5, 5, 50)

    # calculate output
    y = x*theta[0] + x**2*theta[1] + 0.5 + 0.2*np.random.randn(50)

    # return data
    return np.expand_dims(x, axis=1), y

def ex13_generate_data():

    # cluster 1
    X1 = [[-3], [4]]*np.ones((1,25)) + np.random.randn(2,25)

    # cluster 2
    X2 = [[2], [1]]*np.ones((1,25)) + np.random.randn(2,25)

    # set y matrix 
    y = np.append(np.zeros(25), np.ones(25))

    # shuffle data
    X = np.append(X1, X2, axis=1)
    perm = np.random.permutation(X.shape[1])
    X = X[:,perm].T
    y = y[perm]

    # return data
    return X, y