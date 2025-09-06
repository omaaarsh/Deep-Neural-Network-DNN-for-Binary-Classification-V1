"""
Cost function implementations for neural networks
"""
import numpy as np


class CostFunction:
    """Class for cost function computations"""
    
    @staticmethod
    def compute_cost(AL, Y):
        """
        Implement the cost function (cross-entropy)

        Arguments:
        AL -- probability vector corresponding to label predictions, shape (1, number of examples)
        Y -- true "label" vector, shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
        m = Y.shape[1]
        
        # Compute cross-entropy cost
        cost = (1./m) * (-np.dot(Y, np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        cost = np.squeeze(cost)  # Remove unnecessary dimensions
        assert(cost.shape == ())
        
        return cost
    
    @staticmethod
    def compute_cost_with_regularization(AL, Y, parameters, lambd):
        """
        Implement the cost function with L2 regularization
        
        Arguments:
        AL -- probability vector corresponding to label predictions
        Y -- true "label" vector
        parameters -- python dictionary containing parameters of the model
        lambd -- regularization hyperparameter
        
        Returns:
        cost -- cross-entropy cost with L2 regularization
        """
        m = Y.shape[1]
        
        # Cross-entropy cost
        cross_entropy_cost = CostFunction.compute_cost(AL, Y)
        
        # L2 regularization cost
        L2_regularization_cost = 0
        L = len(parameters) // 2  # number of layers
        
        for l in range(1, L + 1):
            L2_regularization_cost += np.sum(np.square(parameters['W' + str(l)]))
        
        L2_regularization_cost = (lambd / (2 * m)) * L2_regularization_cost
        
        cost = cross_entropy_cost + L2_regularization_cost
        
        return cost