# load data
from IPython.display import clear_output
from tensorflow import keras
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from tensorflow.keras.datasets import mnist


class customProgressBar(keras.callbacks.Callback):
    # callbacks = [ProgressStatus()]
    def on_train_begin(self, logs={}):
        self.total_epochs = self.params['epochs']
        self.current_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        self.current_epoch += 1
        clear_output(wait=True)
        print(f'epoch {self.current_epoch}/{self.total_epochs}')

def ex211_generate_data():

    # specify \theta 
    theta = 0.25

    # generate some input data
    x = np.random.uniform(-5, 5, 50)

    # calculate output
    y = x*theta + 0.5 + 0.2*np.random.randn(50)

    # return data
    return np.expand_dims(x, axis=1), y

def ex212_generate_data():

    # specify \theta 
    theta = [0.25, -0.1]

    # generate some input data
    x = np.random.uniform(-5, 5, 50)

    # calculate output
    y = x*theta[0] + x**2*theta[1] + 0.5 + 0.2*np.random.randn(50)

    # return data
    return np.expand_dims(x, axis=1), y

def ex23_load_MNIST():

    # get MNIST data set and just keep the smaller test set
    (X_train, y_train), (_, _) = mnist.load_data()

    # return the X values of the test set
    return X_train, y_train
    

def ex23_plot_MNIST_impression(X):
    
    # create axes
    _, ax = plt.subplots(ncols=3, nrows=3)

    # plot data in axes
    for i in range(9):  
        ax[i%3, i//3].imshow(X[i], cmap=plt.get_cmap('gray'))

    return