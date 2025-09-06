"""
Main script to demonstrate neural network usage
"""
import numpy as np
from data_loader import DataLoader
from neural_network import NeuralNetwork
from visualizer import Visualizer


def main():
    """Main function to run the neural network example"""
    
    print("=" * 60)
    print("DEEP LEARNING NEURAL NETWORK - CAT vs NON-CAT CLASSIFICATION")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    try:
        train_x_orig, train_y, test_x_orig, test_y, classes = DataLoader.load_data()
        train_x, test_x = DataLoader.preprocess_data(train_x_orig, test_x_orig)
        
        print(f"   Training set: {train_x.shape[1]} examples")
        print(f"   Test set: {test_x.shape[1]} examples")
        print(f"   Image size: {train_x_orig.shape[1]}x{train_x_orig.shape[2]}x{train_x_orig.shape[3]}")
        print(f"   Classes: {classes}")
        
    except FileNotFoundError:
        print("   Warning: H5 files not found. Using dummy data for demonstration.")
        # Create dummy data for demonstration
        train_x = np.random.randn(12288, 200)  # 64*64*3 = 12288 features
        train_y = np.random.randint(0, 2, (1, 200))
        test_x = np.random.randn(12288, 50)
        test_y = np.random.randint(0, 2, (1, 50))
        classes = np.array([b'non-cat', b'cat'])
    
    # Define network architecture
    print("\n2. Defining network architecture...")
    layer_dims = [train_x.shape[0], 20, 7, 5, 1]  # 4-layer model
    print(f"   Architecture: {layer_dims}")
    
    # Create and configure neural network
    print("\n3. Creating neural network...")
    nn = NeuralNetwork(layer_dims, initialization_method='he')
    
    # Training parameters
    learning_rate = 0.0075
    num_iterations = 2500
    
    print(f"   Learning rate: {learning_rate}")
    print(f"   Number of iterations: {num_iterations}")
    print(f"   Initialization method: He initialization")
    
    # Train the model
    print("\n4. Training the model...")
    print("   (This may take a few moments...)")
    
    parameters = nn.train(train_x, train_y, 
                         learning_rate=learning_rate, 
                         num_iterations=num_iterations, 
                         print_cost=True)
    
    # Make predictions
    print("\n5. Making predictions...")
    print("\n   Training set predictions:")
    train_predictions = nn.predict(train_x, train_y)
    
    print("\n   Test set predictions:")
    test_predictions = nn.predict(test_x, test_y)
    
    # Visualize results
    print("\n6. Visualizing results...")
    
    # Plot cost function
    if nn.get_costs():
        Visualizer.plot_costs(nn.get_costs(), title="Training Cost Over Time")
    
    # Plot weight distribution for first layer
    Visualizer.plot_weights_histogram(nn.get_parameters(), layer_num=1)
    
    # Display some statistics
    print("\n7. Model Summary:")
    print("   " + "="*40)
    total_params = sum(param.size for param in nn.get_parameters().values())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Final training cost: {nn.get_costs()[-1]:.6f}" if nn.get_costs() else "   No cost history available")
    
    # Calculate and display final accuracies
    if len(train_predictions) > 0 and len(test_predictions) > 0:
        train_accuracy = np.mean(train_predictions == train_y)
        test_accuracy = np.mean(test_predictions == test_y)
        print(f"   Final training accuracy: {train_accuracy:.4f}")
        print(f"   Final test accuracy: {test_accuracy:.4f}")
    
    print("\n   Training completed successfully!")
    print("=" * 60)
    
    return nn, train_predictions, test_predictions


def demonstrate_regularization():
    """Demonstrate neural network with regularization"""
    
    print("\n" + "="*60)
    print("REGULARIZATION DEMONSTRATION")
    print("="*60)
    
    # Create dummy data
    train_x = np.random.randn(100, 200) * 0.5
    train_y = np.random.randint(0, 2, (1, 200))
    
    layer_dims = [100, 50, 20, 1]
    
    # Train without regularization
    print("\n1. Training without regularization...")
    nn_no_reg = NeuralNetwork(layer_dims)
    nn_no_reg.train(train_x, train_y, learning_rate=0.01, num_iterations=1000, print_cost=True)
    
    # Train with regularization
    print("\n2. Training with L2 regularization (lambda=0.7)...")
    nn_reg = NeuralNetwork(layer_dims)
    nn_reg.train(train_x, train_y, learning_rate=0.01, num_iterations=1000, print_cost=True, lambd=0.7)
    
    # Compare results
    print("\n3. Comparing results...")
    Visualizer.plot_learning_curves(nn_no_reg.get_costs(), nn_reg.get_costs())


if __name__ == "__main__":
    # Run main demonstration
    model, train_preds, test_preds = main()
    
    # Ask user if they want to see regularization demo
    try:
        response = input("\nWould you like to see the regularization demonstration? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            demonstrate_regularization()
    except (KeyboardInterrupt, EOFError):
        print("\nDemo completed.")
    
    print("\nAll demonstrations completed!")