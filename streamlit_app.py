"""
Streamlit app for Neural Network visualization and education
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
import time

# Import our neural network classes
try:
    from neural_network import NeuralNetwork
    from data_loader import DataLoader
    from visualizer import Visualizer
    from activation_functions import ActivationFunctions
    from cost_function import CostFunction
except ImportError:
    st.error("Please make sure all neural network modules are in the same directory!")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Neural Network Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        border-left: 4px solid #ff7f0e;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Main header
    st.markdown('<h1 class="main-header">🧠 Neural Network Explorer</h1>', unsafe_allow_html=True)
    st.markdown("**An Interactive Guide to Deep Learning**")
    
    # Sidebar for navigation and controls
    st.sidebar.title("🎛️ Control Panel")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose a section:",
        [
            "📚 Introduction to Neural Networks",
            "🏗️ Network Architecture",
            "⚙️ Training Process",
            "🔍 Step-by-Step Analysis",
            "🐱 Cat vs Non-Cat Example"   # 🔥 New page
        ]
    )
    
    if page == "📚 Introduction to Neural Networks":
        show_introduction()
    elif page == "🏗️ Network Architecture":
        show_architecture()
    elif page == "⚙️ Training Process":
        show_training_process()
    elif page == "🔍 Step-by-Step Analysis":
        show_step_analysis()
    elif page == "🐱 Cat vs Non-Cat Example":
        show_cat_vs_non_cat()  # 🔥 Call the new function

def show_introduction():
    """Introduction page with neural network basics"""
    
    st.markdown('<div class="step-header">What is a Neural Network?</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        A neural network is a computational model inspired by biological neural networks. It consists of:
        
        • **Neurons (Nodes)**: Basic processing units that receive inputs and produce outputs
        • **Connections (Weights)**: Links between neurons that have associated strengths
        • **Layers**: Groups of neurons that process information at different levels
        • **Activation Functions**: Mathematical functions that determine neuron output
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Simple network visualization
        fig = create_simple_network_diagram()
        st.plotly_chart(fig, use_container_width=True)
    
    # Interactive activation function demo
    st.markdown('<div class="step-header">Activation Functions</div>', unsafe_allow_html=True)
    
    activation_type = st.selectbox("Choose an activation function:", ["ReLU", "Sigmoid", "Tanh"])
    
    x = np.linspace(-10, 10, 100)
    
    if activation_type == "ReLU":
        y = np.maximum(0, x)
        description = "ReLU (Rectified Linear Unit) outputs the input directly if positive, otherwise zero."
    elif activation_type == "Sigmoid":
        y = 1 / (1 + np.exp(-x))
        description = "Sigmoid function squashes input to a range between 0 and 1."
    else:  # Tanh
        y = np.tanh(x)
        description = "Tanh function squashes input to a range between -1 and 1."
    
    fig = px.line(x=x, y=y, title=f"{activation_type} Activation Function")
    fig.update_layout(xaxis_title="Input", yaxis_title="Output")
    st.plotly_chart(fig, use_container_width=True)
    st.info(description)

def show_architecture():
    """Show network architecture visualization"""
    
    st.markdown('<div class="step-header">Design Your Network Architecture</div>', unsafe_allow_html=True)
    
    # Architecture controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Network Configuration")
        
        input_size = st.number_input("Input Layer Size", min_value=1, max_value=1000, value=784)
        
        num_hidden = st.number_input("Number of Hidden Layers", min_value=1, max_value=5, value=2)
        
        hidden_sizes = []
        for i in range(num_hidden):
            size = st.number_input(f"Hidden Layer {i+1} Size", min_value=1, max_value=500, value=64-i*16)
            hidden_sizes.append(size)
        
        output_size = st.number_input("Output Layer Size", min_value=1, max_value=10, value=1)
        
        layer_dims = [input_size] + hidden_sizes + [output_size]
    
    with col2:
        st.subheader("Network Visualization")
        fig = create_network_architecture_plot(layer_dims)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate parameters
        total_params = calculate_total_parameters(layer_dims)
        st.markdown(f"""
        <div class="success-box">
        <strong>Network Summary:</strong><br>
        • Total Layers: {len(layer_dims)}<br>
        • Total Parameters: {total_params:,}<br>
        • Architecture: {' → '.join(map(str, layer_dims))}
        </div>
        """, unsafe_allow_html=True)

def show_training_process():
    """Explain the training process step by step"""
    
    st.markdown('<div class="step-header">Neural Network Training Process</div>', unsafe_allow_html=True)
    
    # Training steps explanation
    steps = [
        ("1. Forward Propagation", "Data flows forward through the network to generate predictions"),
        ("2. Cost Calculation", "Compare predictions with true labels using a cost function"),
        ("3. Backward Propagation", "Calculate gradients by propagating errors backward"),
        ("4. Parameter Update", "Update weights and biases using gradient descent"),
        ("5. Repeat", "Continue until convergence or maximum iterations")
    ]
    
    selected_step = st.selectbox("Select a step to explore:", [step[0] for step in steps])
    
    # Find selected step details
    step_details = next((step for step in steps if step[0] == selected_step), None)
    
    if step_details:
        st.markdown(f"### {step_details[0]}")
        st.info(step_details[1])
        
        # Show mathematical details based on selected step
        if "Forward Propagation" in selected_step:
            show_forward_propagation_math()
        elif "Cost Calculation" in selected_step:
            show_cost_calculation_math()
        elif "Backward Propagation" in selected_step:
            show_backward_propagation_math()
        elif "Parameter Update" in selected_step:
            show_parameter_update_math()

def show_step_analysis():
    """Detailed step-by-step analysis of a single forward and backward pass"""
    
    st.markdown('<div class="step-header">Step-by-Step Network Analysis</div>', unsafe_allow_html=True)
    
    # Create a simple network for analysis
    layer_dims = [4, 3, 2, 1]  # Simple network for clarity
    
    # Generate sample input
    np.random.seed(42)
    sample_input = np.random.randn(4, 1)
    
    st.subheader("Sample Network: 4 → 3 → 2 → 1")
    
    # Initialize network
    if 'analysis_network' not in st.session_state:
        st.session_state.analysis_network = NeuralNetwork(layer_dims)
    
    nn = st.session_state.analysis_network
    
    # Step-by-step forward pass
    st.subheader("Forward Pass Analysis")
    
    A = sample_input
    st.write(f"Input: {A.flatten()}")
    
    # Show each layer computation
    caches = []
    for i in range(1, len(layer_dims)):
        W = nn.parameters[f'W{i}']
        b = nn.parameters[f'b{i}']
        
        Z = np.dot(W, A) + b
        
        if i < len(layer_dims) - 1:  # Hidden layers use ReLU
            A_new, cache = ActivationFunctions.relu(Z)
            activation_name = "ReLU"
        else:  # Output layer uses sigmoid
            A_new, cache = ActivationFunctions.sigmoid(Z)
            activation_name = "Sigmoid"
        
        # Store for visualization
        caches.append((A, W, b, Z, A_new))
        
        with st.expander(f"Layer {i} Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("Weight Matrix (W):")
                st.dataframe(pd.DataFrame(W))
                st.write("Bias Vector (b):")
                st.dataframe(pd.DataFrame(b))
            
            with col2:
                st.write(f"Linear Output (Z = WA + b):")
                st.dataframe(pd.DataFrame(Z))
                st.write(f"After {activation_name} Activation:")
                st.dataframe(pd.DataFrame(A_new))
        
        A = A_new
    
    st.success(f"Final Output: {A.flatten()[0]:.6f}")

def create_simple_network_diagram():
    """Create a simple network diagram using Plotly"""
    
    fig = go.Figure()
    
    # Node positions
    layers = [3, 4, 2]  # nodes per layer
    x_positions = [0, 1, 2]
    
    # Add connections
    for i in range(len(layers)-1):
        for j in range(layers[i]):
            for k in range(layers[i+1]):
                y1 = j - (layers[i]-1)/2
                y2 = k - (layers[i+1]-1)/2
                fig.add_trace(go.Scatter(
                    x=[x_positions[i], x_positions[i+1]],
                    y=[y1, y2],
                    mode='lines',
                    line=dict(color='lightgray', width=1),
                    showlegend=False,
                    hoverinfo='none'
                ))
    
    # Add nodes
    for i, layer_size in enumerate(layers):
        for j in range(layer_size):
            y_pos = j - (layer_size-1)/2
            fig.add_trace(go.Scatter(
                x=[x_positions[i]],
                y=[y_pos],
                mode='markers',
                marker=dict(size=20, color=['lightblue', 'orange', 'lightgreen'][i]),
                showlegend=False,
                hoverinfo='none'
            ))
    
    fig.update_layout(
        title="Simple Neural Network",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_network_architecture_plot(layer_dims):
    """Create network architecture visualization"""
    
    fig = go.Figure()
    
    max_nodes = max(layer_dims)
    x_positions = list(range(len(layer_dims)))
    
    # Add connections between layers
    for i in range(len(layer_dims)-1):
        current_layer_size = layer_dims[i]
        next_layer_size = layer_dims[i+1]
        
        for j in range(current_layer_size):
            for k in range(next_layer_size):
                y1 = j - (current_layer_size-1)/2
                y2 = k - (next_layer_size-1)/2
                
                fig.add_trace(go.Scatter(
                    x=[x_positions[i], x_positions[i+1]],
                    y=[y1, y2],
                    mode='lines',
                    line=dict(color='lightgray', width=0.5),
                    showlegend=False,
                    hoverinfo='none'
                ))
    
    # Add nodes
    colors = ['lightblue', 'orange', 'lightcoral', 'lightgreen', 'plum', 'khaki']
    for i, layer_size in enumerate(layer_dims):
        for j in range(layer_size):
            y_pos = j - (layer_size-1)/2
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=[x_positions[i]],
                y=[y_pos],
                mode='markers',
                marker=dict(size=15, color=color, line=dict(width=2, color='white')),
                showlegend=False,
                hovertext=f"Layer {i+1}, Node {j+1}",
                hoverinfo='text'
            ))
    
    # Add layer labels
    layer_names = ['Input'] + [f'Hidden {i}' for i in range(1, len(layer_dims)-1)] + ['Output']
    for i, (pos, name, size) in enumerate(zip(x_positions, layer_names, layer_dims)):
        fig.add_annotation(
            x=pos, y=-(max_nodes/2 + 1),
            text=f"{name}<br>({size} nodes)",
            showarrow=False,
            font=dict(size=10)
        )
    
    fig.update_layout(
        title="Network Architecture",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        margin=dict(l=0, r=0, t=40, b=60)
    )
    
    return fig

def calculate_total_parameters(layer_dims):
    """Calculate total number of parameters in the network"""
    total = 0
    for i in range(1, len(layer_dims)):
        # Weights: current_layer_size × previous_layer_size
        weights = layer_dims[i] * layer_dims[i-1]
        # Biases: current_layer_size
        biases = layer_dims[i]
        total += weights + biases
    return total

def show_forward_propagation_math():
    """Show forward propagation mathematical details"""
    st.markdown("""
    ### Forward Propagation Mathematics
    
    For each layer l, the computation involves:
    
    **Linear Transformation:**
    ```
    Z[l] = W[l] × A[l-1] + b[l]
    ```
    
    **Activation Function:**
    ```
    A[l] = g(Z[l])
    ```
    
    Where:
    - W[l]: Weight matrix for layer l
    - b[l]: Bias vector for layer l  
    - A[l-1]: Activations from previous layer
    - g(): Activation function (ReLU, Sigmoid, etc.)
    """)

def show_cost_calculation_math():
    """Show cost calculation mathematical details"""
    st.markdown("""
    ### Cost Function Mathematics
    
    **Binary Cross-Entropy Cost:**
    ```
    J = -(1/m) × Σ[y⁽ⁱ⁾ × log(ŷ⁽ⁱ⁾) + (1-y⁽ⁱ⁾) × log(1-ŷ⁽ⁱ⁾)]
    ```
    
    **With L2 Regularization:**
    ```
    J = J₀ + (λ/2m) × Σ||W[l]||²
    ```
    
    Where:
    - m: Number of training examples
    - y⁽ⁱ⁾: True label for example i
    - ŷ⁽ⁱ⁾: Predicted probability for example i
    - λ: Regularization parameter
    """)

def show_backward_propagation_math():
    """Show backward propagation mathematical details"""
    st.markdown("""
    ### Backward Propagation Mathematics
    
    **Output Layer Gradients:**
    ```
    dA[L] = -(y/ŷ - (1-y)/(1-ŷ))
    dZ[L] = dA[L] × g'(Z[L])
    ```
    
    **Hidden Layer Gradients:**
    ```
    dA[l] = W[l+1]ᵀ × dZ[l+1]
    dZ[l] = dA[l] × g'(Z[l])
    ```
    
    **Parameter Gradients:**
    ```
    dW[l] = (1/m) × dZ[l] × A[l-1]ᵀ
    db[l] = (1/m) × Σ dZ[l]
    ```
    """)

def show_parameter_update_math():
    """Show parameter update mathematical details"""
    st.markdown("""
    ### Parameter Update Mathematics
    
    **Gradient Descent Update Rule:**
    ```
    W[l] = W[l] - α × dW[l]
    b[l] = b[l] - α × db[l]
    ```
    
    Where:
    - α: Learning rate
    - dW[l], db[l]: Gradients computed during backpropagation
    
    **Learning Rate Guidelines:**
    - Too small: Slow convergence
    - Too large: Overshooting, instability
    - Typical range: 0.001 - 0.1
    """)
def show_function_approximation():
    """Function approximation playground"""
    st.write("See how neural networks can approximate complex mathematical functions!")
    
    st.markdown("""
    This section would demonstrate:
    - Universal approximation theorem
    - Effect of network depth and width
    - Overfitting vs underfitting
    - Regularization effects
    """)
    
    st.info("Function approximation demo - coming soon!")

def show_cat_vs_non_cat():
    """Run the Cat vs Non-Cat neural network example"""

    st.markdown('<div class="step-header">🐱 Cat vs Non-Cat Classification</div>', unsafe_allow_html=True)
    st.info("This demo trains a neural network on the classic Cat vs Non-Cat dataset.")

    # ---------- 1. Load and preprocess data ----------
    try:
        train_x_orig, train_y, test_x_orig, test_y, classes = DataLoader.load_data()
        train_x, test_x = DataLoader.preprocess_data(train_x_orig, test_x_orig)

        st.success(f"Training set: {train_x.shape[1]} examples")
        st.success(f"Test set: {test_x.shape[1]} examples")
        st.success(f"Image size: {train_x_orig.shape[1]}x{train_x_orig.shape[2]}x{train_x_orig.shape[3]}")
        st.success(f"Classes: {classes}")

        # --- Show sample images ---
        st.write("### Example Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(train_x_orig[0], caption=f"Label: {classes[train_y[0][0]].decode('utf-8')}")
        with col2:
            st.image(train_x_orig[1], caption=f"Label: {classes[train_y[0][1]].decode('utf-8')}")

    except FileNotFoundError:
        st.warning("⚠️ H5 files not found. Using dummy data for demo.")
        train_x = np.random.randn(12288, 200)
        train_y = np.random.randint(0, 2, (1, 200))
        test_x = np.random.randn(12288, 50)
        test_y = np.random.randint(0, 2, (1, 50))
        classes = np.array([b'non-cat', b'cat'])

    # ---------- 2. Define network ----------
    st.write("### 2. Defining network architecture...")
    layer_dims = [train_x.shape[0], 20, 7, 5, 1]
    st.info(f"Architecture: {layer_dims}")

    nn = NeuralNetwork(layer_dims, initialization_method='he')

    learning_rate = 0.0075
    num_iterations = 2000
    st.write(f"Learning rate: {learning_rate}, Iterations: {num_iterations}")

    # ---------- 3. Train with progress ----------
    st.write("### 3. Training the model...")
    progress = st.progress(0)
    for i in range(num_iterations):
        nn.train(train_x, train_y, learning_rate=learning_rate, num_iterations=1, print_cost=False)
        if i % 100 == 0:
            progress.progress(int(i / num_iterations * 100))

    st.success("🎉 Training completed!")

    # ---------- 4. Evaluate ----------
    train_pred = nn.predict(train_x)
    test_pred = nn.predict(test_x)

    train_acc = np.mean(train_pred == train_y)
    test_acc = np.mean(test_pred == test_y)

    st.metric("Training Accuracy", f"{train_acc:.4f}")
    st.metric("Test Accuracy", f"{test_acc:.4f}")

    # ---------- 5. Upload or select image for prediction ----------
    st.write("### 4. Try it yourself!")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).resize((64, 64))
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_array = np.array(img).reshape(-1, 1) / 255.
        prediction = nn.predict(img_array)

        label = "Cat 😺" if prediction[0][0] == 1 else "Non-Cat 🐶"
        st.success(f"Prediction: {label}")

    # ---------- 6. Visualize NN Architecture ----------
    st.write("### 5. Neural Network Visualization 🧠")
    graph = f"""
    digraph G {{
        rankdir=LR;
        Input [label="Input Layer ({layer_dims[0]} features)", shape=box, style=filled, color=lightblue];
        Hidden1 [label="Hidden Layer 1 ({layer_dims[1]} units)", shape=ellipse, style=filled, color=lightgreen];
        Hidden2 [label="Hidden Layer 2 ({layer_dims[2]} units)", shape=ellipse, style=filled, color=lightgreen];
        Hidden3 [label="Hidden Layer 3 ({layer_dims[3]} units)", shape=ellipse, style=filled, color=lightgreen];
        Output [label="Output Layer ({layer_dims[4]} unit)", shape=box, style=filled, color=orange];

        Input -> Hidden1 -> Hidden2 -> Hidden3 -> Output;
    }}
    """
    st.graphviz_chart(graph)

    # ---------- 7. Neural Network Math ----------
    st.write("### 6. Neural Network Mathematics 🧮")
    st.markdown("""
    **Forward Propagation:**  
    Z⁽ˡ⁾ = W⁽ˡ⁾A⁽ˡ⁻¹⁾ + b⁽ˡ⁾  
    A⁽ˡ⁾ = g(Z⁽ˡ⁾)

    **Cost Function (Binary Cross-Entropy):**  
    J = -(1/m) Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

    **Backward Propagation:**  
    dZ⁽ᴸ⁾ = A⁽ᴸ⁾ - Y  
    dW⁽ˡ⁾ = (1/m) dZ⁽ˡ⁾ A⁽ˡ⁻¹⁾ᵀ  
    db⁽ˡ⁾ = (1/m) Σ dZ⁽ˡ⁾  

    **Gradient Descent:**  
    W⁽ˡ⁾ = W⁽ˡ⁾ - α dW⁽ˡ⁾  
    b⁽ˡ⁾ = b⁽ˡ⁾ - α db⁽ˡ⁾
    """)
if __name__ == "__main__":
    main()