"""
Visualization utilities for neural networks
"""
import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    """Class for visualization utilities"""
    
    @staticmethod
    def print_mislabeled_images(classes, X, y, p, max_images=10):
        """
        Plots images where predictions and truth were different.
        
        Arguments:
        classes -- list of class names
        X -- dataset
        y -- true labels
        p -- predictions
        max_images -- maximum number of images to display
        """
        a = p + y
        mislabeled_indices = np.asarray(np.where(a == 1))
        
        if len(mislabeled_indices[0]) == 0:
            print("No mislabeled images found!")
            return
        
        num_images = min(len(mislabeled_indices[0]), max_images)
        
        if num_images > 0:
            plt.figure(figsize=(15, 8))
            
            for i in range(num_images):
                index = mislabeled_indices[1][i]
                
                plt.subplot(2, (num_images + 1) // 2, i + 1)
                plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
                plt.axis('off')
                
                pred_class = classes[int(p[0, index])].decode("utf-8") if hasattr(classes[int(p[0, index])], 'decode') else str(classes[int(p[0, index])])
                true_class = classes[y[0, index]].decode("utf-8") if hasattr(classes[y[0, index]], 'decode') else str(classes[y[0, index]])
                
                plt.title(f"Prediction: {pred_class}\nActual: {true_class}")
            
            plt.tight_layout()
            plt.show()
    
    @staticmethod
    def plot_costs(costs, title="Cost Function", xlabel="Iterations (per hundreds)", ylabel="Cost"):
        """
        Plot the cost function over iterations
        
        Arguments:
        costs -- list of cost values
        title -- plot title
        xlabel -- x-axis label
        ylabel -- y-axis label
        """
        plt.figure(figsize=(10, 6))
        plt.plot(costs)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_learning_curves(train_costs, val_costs=None):
        """
        Plot training and validation learning curves
        
        Arguments:
        train_costs -- training cost history
        val_costs -- validation cost history (optional)
        """
        plt.figure(figsize=(12, 5))
        
        # Plot costs
        plt.subplot(1, 2, 1)
        plt.plot(train_costs, label='Training Cost')
        if val_costs is not None:
            plt.plot(val_costs, label='Validation Cost')
        plt.xlabel('Iterations (per hundreds)')
        plt.ylabel('Cost')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        
        # Plot cost difference if validation costs are provided
        if val_costs is not None:
            plt.subplot(1, 2, 2)
            cost_diff = np.array(val_costs) - np.array(train_costs)
            plt.plot(cost_diff, color='red')
            plt.xlabel('Iterations (per hundreds)')
            plt.ylabel('Validation - Training Cost')
            plt.title('Overfitting Detection')
            plt.grid(True)
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_decision_boundary(X, y, model, h=0.01):
        """
        Plot decision boundary for 2D data
        
        Arguments:
        X -- 2D input data
        y -- labels
        model -- trained neural network model
        h -- step size in the mesh
        """
        # Get data bounds
        x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
        
        # Create mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Make predictions on mesh points
        mesh_points = np.c_[xx.ravel(), yy.ravel()].T
        Z = model.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        
        # Plot data points
        scatter = plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.RdYlBu, edgecolors='black')
        plt.colorbar(scatter)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')
        plt.show()
    