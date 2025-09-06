"""
Layer operations for neural networks
"""
import numpy as np
from activation_functions import ActivationFunctions


class Layer:
    """Class representing a single layer in the neural network"""
    
    @staticmethod
    def linear_forward(A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- a python dictionary containing "A", "W" and "b"
        """
        Z = W.dot(A) + b
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        return Z, cache

    @staticmethod
    def linear_activation_forward(A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer
        W -- weights matrix
        b -- bias vector
        activation -- the activation to be used: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function
        cache -- stored for computing the backward pass efficiently
        """
        if activation == "sigmoid":
            Z, linear_cache = Layer.linear_forward(A_prev, W, b)
            A, activation_cache = ActivationFunctions.sigmoid(Z)
        elif activation == "relu":
            Z, linear_cache = Layer.linear_forward(A_prev, W, b)
            A, activation_cache = ActivationFunctions.relu(Z)
        else:
            raise ValueError("Error! Please make sure you have passed the value correctly in the \"activation\" parameter")
        
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)
        return A, cache

    @staticmethod
    def linear_backward(dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output
        cache -- tuple of values (A_prev, W, b) from forward propagation

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer)
        dW -- Gradient of the cost with respect to W (current layer)
        db -- Gradient of the cost with respect to b (current layer)
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1./m * np.dot(dZ, A_prev.T)
        db = 1./m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db

    @staticmethod
    def linear_activation_backward(dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache)
        activation -- the activation to be used: "sigmoid" or "relu"
        
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer)
        dW -- Gradient of the cost with respect to W (current layer)
        db -- Gradient of the cost with respect to b (current layer)
        """
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            dZ = ActivationFunctions.relu_backward(dA, activation_cache)
            dA_prev, dW, db = Layer.linear_backward(dZ, linear_cache)
        elif activation == "sigmoid":
            dZ = ActivationFunctions.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = Layer.linear_backward(dZ, linear_cache)
        else:
            raise ValueError("Error! Please make sure you have passed the value correctly in the \"activation\" parameter")
        
        return dA_prev, dW, db