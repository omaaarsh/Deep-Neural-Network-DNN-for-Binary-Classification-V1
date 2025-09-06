"""
Main Neural Network class
"""
import numpy as np
from parameter_init import ParameterInitializer
from layer import Layer
from cost_function import CostFunction


class NeuralNetwork:
    """Main Neural Network class that orchestrates the training process"""
    
    def __init__(self, layer_dims, initialization_method='he'):
        """
        Initialize the neural network
        
        Arguments:
        layer_dims -- list containing the dimensions of each layer
        initialization_method -- 'he', 'xavier', or 'random'
        """
        self.layer_dims = layer_dims
        self.initialization_method = initialization_method
        
        # Initialize parameters based on the chosen method
        if initialization_method == 'he':
            self.parameters = ParameterInitializer.initialize_parameters_deep(layer_dims)
        elif initialization_method == 'xavier':
            self.parameters = ParameterInitializer.initialize_parameters_xavier(layer_dims)
        else:
            self.parameters = ParameterInitializer.initialize_parameters_deep(layer_dims)
            
        self.costs = []
    
    def forward_propagation(self, X):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        
        Returns:
        AL -- last post-activation value
        caches -- list of caches for backpropagation
        """
        caches = []
        A = X
        L = len(self.parameters) // 2
        
        # Implement [LINEAR -> RELU]*(L-1)
        for l in range(1, L):
            A_prev = A 
            A, cache = Layer.linear_activation_forward(A_prev, self.parameters['W' + str(l)], 
                                                     self.parameters['b' + str(l)], activation="relu")
            caches.append(cache)
        
        # Implement LINEAR -> SIGMOID
        AL, cache = Layer.linear_activation_forward(A, self.parameters['W' + str(L)], 
                                                   self.parameters['b' + str(L)], activation="sigmoid")
        caches.append(cache)
        
        assert(AL.shape == (1, X.shape[1]))
        return AL, caches
    
    def backward_propagation(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation
        Y -- true "label" vector
        caches -- list of caches from forward propagation
        
        Returns:
        grads -- A dictionary with the gradients
        """
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        
        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        # Lth layer (SIGMOID -> LINEAR) gradients
        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = \
            Layer.linear_activation_backward(dAL, current_cache, activation="sigmoid")
        
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = \
                Layer.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation="relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads
    
    def backward_propagation_with_regularization(self, AL, Y, caches, lambd):
        """
        Implement backward propagation with L2 regularization
        
        Arguments:
        AL -- probability vector, output of forward propagation
        Y -- true "label" vector
        caches -- list of caches from forward propagation  
        lambd -- regularization hyperparameter
        
        Returns:
        grads -- A dictionary with the gradients including regularization
        """
        grads = self.backward_propagation(AL, Y, caches)
        m = AL.shape[1]
        L = len(caches)
        
        # Add regularization to weight gradients
        for l in range(1, L + 1):
            grads["dW" + str(l)] = grads["dW" + str(l)] + (lambd / m) * self.parameters["W" + str(l)]
        
        return grads
    
    def update_parameters(self, grads, learning_rate):
        """
        Update parameters using gradient descent
        
        Arguments:
        grads -- python dictionary containing gradients
        learning_rate -- learning rate for gradient descent
        """
        L = len(self.parameters) // 2
        
        for l in range(L):
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    
    def train(self, X, Y, learning_rate=0.0075, num_iterations=3000, print_cost=False, lambd=0):
        """
        Train the neural network
        
        Arguments:
        X -- training data
        Y -- training labels
        learning_rate -- learning rate for optimization
        num_iterations -- number of iterations for optimization loop
        print_cost -- if True, print the cost every 100 steps
        lambd -- regularization hyperparameter
        
        Returns:
        parameters -- parameters learnt by the model
        """
        for i in range(0, num_iterations):
            # Forward propagation
            AL, caches = self.forward_propagation(X)
            
            # Compute cost
            if lambd == 0:
                cost = CostFunction.compute_cost(AL, Y)
            else:
                cost = CostFunction.compute_cost_with_regularization(AL, Y, self.parameters, lambd)
            
            # Backward propagation
            if lambd == 0:
                grads = self.backward_propagation(AL, Y, caches)
            else:
                grads = self.backward_propagation_with_regularization(AL, Y, caches, lambd)
            
            # Update parameters
            self.update_parameters(grads, learning_rate)
            
            # Print the cost every 100 training iterations
            if print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
                self.costs.append(cost)
        
        return self.parameters
    
    def predict(self, X, y=None, threshold=0.5):
        """
        Predict the results using the trained model
        
        Arguments:
        X -- data set of examples to label
        y -- true labels (optional, for accuracy calculation)
        threshold -- decision threshold for binary classification
        
        Returns:
        p -- predictions for the given dataset X
        """
        m = X.shape[1]
        p = np.zeros((1, m))
        
        # Forward propagation
        probas, caches = self.forward_propagation(X)
        
        # Convert probas to 0/1 predictions
        p = (probas > threshold).astype(int)
        
        # Print accuracy if labels are provided
        if y is not None:
            accuracy = np.sum((p == y)) / m
            print(f"Accuracy: {accuracy:.4f}")
            
        return p
    
    def get_parameters(self):
        """Return the current parameters"""
        return self.parameters
    
    def set_parameters(self, parameters):
        """Set new parameters"""
        self.parameters = parameters
    
    def get_costs(self):
        """Return the cost history"""
        return self.costs