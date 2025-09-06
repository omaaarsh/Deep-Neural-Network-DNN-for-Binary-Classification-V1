Deep Neural Network (DNN) for Binary Classification V2

Overview
This repository provides a modular, from-scratch implementation of a Deep Neural Network (DNN) for binary classification, designed for educational purposes. It supports multiple hidden layers with ReLU activation, a sigmoid output layer, and gradient descent optimization with optional L2 regularization. The project includes a Streamlit-based interactive web app for visualizing network architecture, training processes, and results, with a focus on the classic cat vs. non-cat image classification dataset.
The codebase is highly modular, enabling easy experimentation, extension, and integration into other projects, while serving as a comprehensive learning tool for deep learning concepts.
Project Structure
neural-network-explorer/
├── activation_functions.py     # ReLU and Sigmoid activation functions and their derivatives
├── cost_function.py            # Cross-entropy loss with optional L2 regularization
├── data_loader.py              # HDF5 data loading and image preprocessing utilities
├── layer.py                    # Forward and backward propagation for linear and activation layers
├── main.py                     # Main script for training, prediction, and regularization demo
├── neural_network.py           # Core NeuralNetwork class (forward/backward prop, training)
├── parameter_init.py           # Initialization strategies (He, Xavier, random)
├── streamlit_app.py            # Streamlit app for interactive visualization and education
├── visualizer.py               # Visualization utilities (costs, weights, decision boundaries)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── *.h5                        # Optional dataset files (train_catvnoncat.h5, test_catvnoncat.h5)

Features

Flexible Architecture: Configurable multi-layer DNN with customizable layer sizes.
Activation Functions: ReLU for hidden layers, Sigmoid for binary output.
Training Enhancements: L2 regularization, He/Xavier initialization, and gradient descent.
Data Handling: Supports HDF5 datasets with preprocessing (flattening, normalization).
Visualization Tools: Plots for training costs, learning curves, weight distributions, and mislabeled images.
Interactive Demo: Streamlit app for exploring network architecture, training steps, and real-time predictions.
Fallback Mechanism: Uses dummy data if dataset files are unavailable.

Network Architecture
Input Layer (n₀)
      ↓
Hidden Layer 1 (n₁) - ReLU
      ↓
Hidden Layer 2 (n₂) - ReLU
      ↓
      ...
      ↓
Output Layer (1) - Sigmoid
      ↓
Binary Prediction (0 or 1)


Input Layer: Raw features (e.g., flattened 64x64x3 images = 12288 features).
Hidden Layers: ReLU activation for non-linear feature extraction.
Output Layer: Sigmoid activation for binary classification probabilities.

Mathematical Foundation
Forward Propagation
For each layer ( l = 1, 2, ..., L ):

Linear Step:[Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}]
Activation Step:
Hidden layers: ( A^{[l]} = \text{ReLU}(Z^{[l]}) = \max(0, Z^{[l]}) )
Output layer: ( A^{[L]} = \sigma(Z^{[L]}) = \frac{1}{1 + e^{-Z^{[L]}}} )



Cost Function

Cross-Entropy Loss:[J = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(A^{L}) + (1 - y^{(i)}) \log(1 - A^{L}) \right]]
With L2 Regularization:[J_{\text{reg}} = J + \frac{\lambda}{2m} \sum_{l=1}^{L} |W^{[l]}|_F^2]where ( \lambda ) is the regularization parameter and ( |W^{[l]}|_F^2 ) is the Frobenius norm of the weight matrix.

Backward Propagation

Output Layer (( l = L )):[dZ^{[L]} = A^{[L]} - Y][dW^{[L]} = \frac{1}{m} dZ^{[L]} \cdot A^{[L-1]T}, \quad db^{[L]} = \frac{1}{m} \sum dZ^{[L]}]
Hidden Layers (( l = L-1, ..., 1 )):[dA^{[l]} = W^{[l+1]T} \cdot dZ^{[l+1]}][dZ^{[l]} = dA^{[l]} \odot g'(Z^{[l]}) \quad (\text{ReLU derivative: } g'(Z) = 1 \text{ if } Z > 0, \text{ else } 0)][dW^{[l]} = \frac{1}{m} dZ^{[l]} \cdot A^{[l-1]T}, \quad db^{[l]} = \frac{1}{m} \sum dZ^{[l]}]
With L2 Regularization:[dW^{[l]} = dW^{[l]} + \frac{\lambda}{m} W^{[l]}]

Parameter Updates
Using gradient descent:[W^{[l]} := W^{[l]} - \alpha \cdot dW^{[l]}, \quad b^{[l]} := b^{[l]} - \alpha \cdot db^{[l]}]where ( \alpha ) is the learning rate.
Training Flow
┌───────────────────────┐
│ Initialize Parameters │
│ W[l], b[l] (He/Xavier)│
└───────────────────────┘
             │
             ▼
┌───────────────────────┐
│ Forward Propagation   │────┐
│ [Linear → ReLU]*(L-1) │    │
│ [Linear → Sigmoid]    │    │
└───────────────────────┘    │
             │               │
             ▼               │
┌───────────────────────┐    │
│ Compute Cost (J)      │    │ Training
│ Optional: L2 Regular. │    │  Loop
└───────────────────────┘    │
             │               │
             ▼               │
┌───────────────────────┐    │
│ Backward Propagation  │    │
│ Compute Gradients     │    │
└───────────────────────┘    │
             │               │
             ▼               │
┌───────────────────────┐    │
│ Update Parameters     │────┘
│ W[l] -= α * dW[l]     │
│ b[l] -= α * db[l]     │
└───────────────────────┘
             │
             ▼
    Converged or Max Iterations?

Core Functions

Parameter Initialization:

Initializes weights using He, Xavier, or random strategies; biases set to zeros.
Output: Dictionary with ( W^{[l]} ) and ( b^{[l]} ) for each layer.


Forward Propagation:

Computes predictions through ( [LINEAR \to RELU] \times (L-1) \to [LINEAR \to SIGMOID] ).
Output: Final predictions ( A^{[L]} ) and caches for backpropagation.


Cost Computation:

Calculates cross-entropy loss, with optional L2 regularization term.
Input: Predictions ( A^{[L]} ), true labels ( Y ), and parameters (if regularized).
Output: Scalar cost value.


Backward Propagation:

Computes gradients for all layers using the chain rule.
Supports L2 regularization for weight gradients.
Output: Gradients dictionary with ( dW^{[l]} ), ( db^{[l]} ), and ( dA^{[l]} ).


Parameter Updates:

Updates weights and biases using gradient descent.
Formula: ( W^{[l]} := W^{[l]} - \alpha \cdot dW^{[l]} ), ( b^{[l]} := b^{[l]} - \alpha \cdot db^{[l]} ).



Network Data Flow

Forward Pass:X (input) → Layer 1 (Z₁, A₁) → Layer 2 (Z₂, A₂) → ... → Layer L (Z_L, A_L) → Output


Backward Pass:dA_L ← dZ[L] ← dW[L], db[L]
     ↑
dA[L-1] ← dZ[L-1] ← dW[L-1], db[L-1]
     ↑
     ...
     ↑
dA[1] ← dZ[1] ← dW[1], db[1]



Model Training Steps

Initialize parameters (He/Xavier for weights, zeros for biases).
Perform forward propagation to compute predictions.
Compute cost using cross-entropy loss (with optional regularization).
Perform backward propagation to calculate gradients.
Update parameters via gradient descent.
Repeat until convergence or maximum iterations reached.

Key Features

✅ Scalable Architecture: Supports any number of layers and neurons.
✅ Efficient Activations: ReLU for hidden layers avoids vanishing gradients; Sigmoid for binary output.
✅ Regularization: L2 regularization to prevent overfitting.
✅ Initialization Options: He, Xavier, or random initialization for stable training.
✅ Vectorized Implementation: Optimized matrix operations using NumPy.
✅ Visualization Support: Cost plots, learning curves, weight histograms, and mislabeled image displays.
✅ Interactive Streamlit App: Educational interface for exploring DNN mechanics and testing predictions.

Installation
Prerequisites

Python 3.8 or higher.
Git (for cloning the repository).

Steps

Clone the repository:git clone https://github.com/your-username/neural-network-explorer.git
cd neural-network-explorer


Install dependencies:pip install -r requirements.txt

requirements.txt content:numpy>=1.19.0
matplotlib>=3.3.0
h5py>=2.10.0
streamlit>=1.28.0
plotly>=5.15.0
pandas>=1.3.0


(Optional) Add dataset:
Place train_catvnoncat.h5 and test_catvnoncat.h5 in the project root.
If missing, the code generates dummy data (12288 features).



Usage
1. Run Main Demo
Train and evaluate the DNN on the cat vs. non-cat dataset:
python main.py


Output: Training progress, accuracy metrics, cost plots, weight histograms, and an optional regularization demo.
Customize: Modify layer_dims, learning_rate, or num_iterations in main.py.

2. Run Streamlit App
Launch the interactive web app:
streamlit run streamlit_app.py


Access at http://localhost:8501.
Explore sections: Introduction, Architecture, Training Process, Step-by-Step Analysis, and Cat vs. Non-Cat Demo.
Upload images for real-time predictions in the cat vs. non-cat section.

3. Regularization Demo
In main.py, respond 'y' to the prompt after the main demo to compare training with and without L2 regularization.
4. Custom Usage
Use the NeuralNetwork class in your own scripts:
from neural_network import NeuralNetwork
import numpy as np

# Define architecture
layer_dims = [12288, 20, 7, 5, 1]  # For 64x64x3 images
nn = NeuralNetwork(layer_dims, initialization_method='he')

# Train
nn.train(X_train, Y_train, learning_rate=0.0075, num_iterations=2500, lambd=0.7)

# Predict
predictions = nn.predict(X_test)

Performance Notes

Learning Rate:
Typical range: 0.001–0.01.
High: Causes oscillations or divergence.
Low: Slow convergence or local minima traps.


Architecture:
Deeper networks learn complex patterns but require careful tuning.
Wider layers increase capacity but may overfit.
Balance depth and width based on data complexity.


Initialization:
He/Xavier initialization prevents vanishing/exploding gradients.
Random initialization with small values as a fallback.


Activations:
ReLU: Fast, prevents vanishing gradients, suitable for deep networks.
Sigmoid: Ideal for binary classification output.


Training Tips:
Monitor cost curves for convergence.
Use regularization (( \lambda \approx 0.7 )) for complex datasets.
Validate on test data to detect overfitting.



Dataset

Cat vs. Non-Cat:
Training: 209 images (64x64x3).
Test: 50 images (64x64x3).
Labels: 1 (cat), 0 (non-cat).
Format: HDF5 via h5py.
Fallback: Dummy data (12288 features) if files are missing.



Contributing

Fork the repository.
Create a feature branch (git checkout -b feature/YourFeature).
Commit changes (git commit -m 'Add YourFeature').
Push to the branch (git push origin feature/YourFeature).
Open a Pull Request with tests and documentation.

License
MIT License. See LICENSE for details.
Acknowledgments

Inspired by Andrew Ng’s Deep Learning Specialization (Coursera).
Built with NumPy, Matplotlib, Streamlit, Plotly, and Pandas.

Star the repo or share feedback to support the project! 🚀
