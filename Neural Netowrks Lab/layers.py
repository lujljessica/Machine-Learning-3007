import numpy as np
import pandas as pd
class Layer:
    """
    Represents a layer in a neural network which can be a input, hidden or output layer
    """
    def __init__(self, n_neurons, n_neurons_next):
        """
        Initializes a new layer
            :param int n_neurons: Current layer size
            :param int n_neurons_next: The number of neurons in this layer.
        """
        self.weights = np.random.rand(n_neurons_next,n_neurons) # num neurosn in next layer is how many rows. Curr layer is columns
        self.activation = None # the activation at CURRENT layer. Can be a array or just float
        self.error = None # stores the error of the layer
        self.gradient = None  # stores the delta values of the layer ie the gradient. Set all to  0 initially
        self.reg_grad = None
        return
    
    def activate(self, x, bias):
        """
        Calculates activation
        """
        # Matrix multiplication / vectorized calculation of activation
        if (bias == True):    
            x_ =  np.insert(x, 0, 1) # inserting a bias at x
        else:
            x_ = x
        self.activation = x_
        z = np.dot(self.weights, np.transpose(x_))
        out = self._apply_sigmoid(z)
        return(out)
    
    def _apply_sigmoid(self, z):
        """
        Applies sigmoid funtcion to all elements in z
        """
        return(1 / (1 + np.exp(-z)))
    
    def apply_sigmoid_derivative(self, a):
        return(a * (1 - a))