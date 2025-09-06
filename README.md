# Deep Neural Network (DNN) for Binary Classification V1

## Overview
This repository implements a Deep Neural Network from scratch for binary classification. The network uses multiple hidden layers with ReLU activation and a sigmoid output layer, trained using gradient descent optimization.
<img width="896" height="320" alt="image" src="https://github.com/user-attachments/assets/3868b020-f946-4886-a2d1-91352e082762" />

## Network Architecture

```
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
Binary Prediction
```

**Architecture Components:**
- **Input Layer**: Raw feature data
- **Hidden Layers**: Feature extraction with ReLU activation
- **Output Layer**: Binary classification with Sigmoid activation

## Mathematical Foundation

### Forward Propagation

For each layer l = 1, 2, ..., L:

**Linear Step:**
```
Z[l] = W[l] × A[l-1] + b[l]
```

**Activation Step:**
- Hidden layers: `A[l] = ReLU(Z[l]) = max(0, Z[l])`
- Output layer: `A[L] = σ(Z[L]) = 1/(1 + e^(-Z[L]))`

### Cost Function

**Cross-Entropy Loss:**
```
J = -(1/m) × Σ[y⁽ⁱ⁾ log(a[L]⁽ⁱ⁾) + (1-y⁽ⁱ⁾) log(1-a[L]⁽ⁱ⁾)]
```

### Backward Propagation

**Output Layer (L):**
```
dZ[L] = A[L] - Y
dW[L] = (1/m) × dZ[L] × A[L-1]ᵀ
db[L] = (1/m) × sum(dZ[L])
```

**Hidden Layers (l = L-1, ..., 1):**
```
dA[l] = W[l+1]ᵀ × dZ[l+1]
dZ[l] = dA[l] ⊙ g'(Z[l])    where g'(Z) = 1 if Z > 0, else 0 for ReLU
dW[l] = (1/m) × dZ[l] × A[l-1]ᵀ
db[l] = (1/m) × sum(dZ[l])
```

**Parameter Updates:**
```
W[l] := W[l] - α × dW[l]
b[l] := b[l] - α × db[l]
```

## Training Flow

```
┌─────────────────┐
│  Initialize     │
│  Parameters     │
│  W[l], b[l]     │
└─────────────────┘
         │
         ▼
    ┌─────────┐
    │ Forward │ ──┐
    │  Pass   │   │
    └─────────┘   │
         │        │
         ▼        │
    ┌─────────┐   │
    │Compute  │   │ Training
    │  Cost   │   │  Loop
    └─────────┘   │
         │        │
         ▼        │
    ┌─────────┐   │
    │Backward │   │
    │  Pass   │   │
    └─────────┘   │
         │        │
         ▼        │
    ┌─────────┐   │
    │ Update  │ ──┘
    │Parameters│
    └─────────┘
         │
         ▼
    Converged?
```

## Core Functions

### 1. Parameter Initialization
- **Purpose**: Initialize network weights and biases
- **Method**: Small random values for weights, zeros for biases
- **Output**: Parameter dictionary with W[l] and b[l] for each layer

### 2. L-Model Forward
- **Purpose**: Complete forward propagation through all layers
- **Process**: [LINEAR → RELU] × (L-1) → LINEAR → SIGMOID
- **Output**: Final predictions and cached values for backprop

### 3. Compute Cost
- **Purpose**: Calculate cross-entropy loss
- **Input**: Predictions and true labels
- **Output**: Scalar cost value

### 4. L-Model Backward
- **Purpose**: Complete backward propagation for all layers
- **Process**: Compute gradients using chain rule from output to input
- **Output**: Gradients dictionary with dW[l] and db[l]

### 5. Update Parameters
- **Purpose**: Apply gradient descent updates
- **Formula**: W := W - α×dW, b := b - α×db
- **Result**: Updated parameters for next iteration

## Network Data Flow

### Forward Pass
```
X (input) → Layer 1 → Layer 2 → ... → Layer L → AL (output)
           Z₁, A₁    Z₂, A₂         ZL, AL
```

### Backward Pass  
```
dAL ← dZ[L] ← dW[L], db[L]
     ↑
dA[L-1] ← dZ[L-1] ← dW[L-1], db[L-1]
     ↑
    ...
     ↑
dA[1] ← dZ[1] ← dW[1], db[1]
```

## Model Training Steps

1. **Initialize Parameters** → Random weights, zero biases
2. **Forward Propagation** → Compute predictions layer by layer  
3. **Cost Computation** → Measure prediction error
4. **Backward Propagation** → Compute gradients via chain rule
5. **Parameter Updates** → Apply gradient descent
6. **Repeat** → Until convergence or max iterations

## Key Features

✅ **Multi-layer Architecture**: Supports any number of hidden layers  
✅ **ReLU Activation**: Prevents vanishing gradients in hidden layers  
✅ **Sigmoid Output**: Probability predictions for binary classification  
✅ **Vectorized Operations**: Efficient matrix computations  
✅ **Gradient Descent**: Iterative parameter optimization  
✅ **Modular Design**: Easy to modify and extend

## Usage Example

```
# Define architecture
layers_dims = [input_size, 20, 7, 5, 1]  # 4-layer network

# Train model
parameters = L_layer_model(X_train, Y_train, layers_dims, 
                          learning_rate=0.0075, 
                          num_iterations=2500)

# Make predictions  
predictions = predict(X_test, parameters)
```

## Performance Notes

- **Learning Rate**: 
  - Range: 0.001 - 0.01 for stable training
  - Too high: Oscillations or divergence
  - Too low: Slow convergence or getting stuck in local minima
  
- **Architecture Design**: 
  - More layers: Can learn more complex patterns but harder to train
  - Layer width: More neurons per layer increases model capacity
  - Balance depth vs width based on data complexity
  
- **Initialization Strategy**: 
  - Proper weight initialization crucial for successful convergence
  - Xavier/He initialization often better than simple random
  - Poor initialization can cause vanishing/exploding gradients
  
- **Activation Choice**: 
  - ReLU prevents vanishing gradients compared to sigmoid/tanh
  - Enables training of deeper networks effectively
  - Computational efficiency: simple max(0,x) operation

- **Training Considerations**:
  - Monitor cost function for convergence patterns
  - Use validation set to detect overfitting
  - Consider regularization techniques for complex datasets
update this
