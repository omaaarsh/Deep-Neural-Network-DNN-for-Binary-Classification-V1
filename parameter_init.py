"""
Parameter initialization strategies for neural networks
"""
import numpy as np


class ParameterInitializer:
    """Class for parameter initialization strategies"""
    
    @staticmethod
    def initialize_parameters(n_x, n_h, n_y):
        """
        Initialize parameters for a 2-layer neural network
        
        Arguments:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer
        
        Returns:
        parameters -- python dictionary containing parameters
        """
        np.random.seed(1)
        
        W1 = np.random.randn(n_h, n_x)*0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h)*0.01
        b2 = np.zeros((n_y, 1))
        
        assert(W1.shape == (n_h, n_x))
        assert(b1.shape == (n_h, 1))
        assert(W2.shape == (n_y, n_h))
        assert(b2.shape == (n_y, 1))
        
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        return parameters

    @staticmethod
    def initialize_parameters_deep(layer_dims):
        """
        Initialize parameters for a deep neural network using He initialization
        
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer
        
        Returns:
        parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL"
        """
        np.random.seed(1)
        parameters = {}
        L = len(layer_dims)

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
            
        return parameters
    
    @staticmethod
    def initialize_parameters_xavier(layer_dims):
        """
        Initialize parameters using Xavier initialization
        
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer
        
        Returns:
        parameters -- python dictionary containing parameters
        """
        np.random.seed(1)
        parameters = {}
        L = len(layer_dims)

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2.0 / layer_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
            
        return parameters