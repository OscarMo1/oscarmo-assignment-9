import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        
        # Activation and gradient storage
        self.hidden_activation = None
        self.output = None

    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation function.")

    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        else:
            raise ValueError("Unsupported activation function.")

    def forward(self, X):
        # Hidden layer
        z1 = np.dot(X, self.W1) + self.b1
        self.hidden_activation = self.activation(z1)
        
        # Output layer
        z2 = np.dot(self.hidden_activation, self.W2) + self.b2
        self.output = 1 / (1 + np.exp(-z2))  # Sigmoid for binary classification
        return self.output

    def backward(self, X, y):
        # Compute gradients
        m = X.shape[0]
        output_error = self.output - y
        dW2 = np.dot(self.hidden_activation.T, output_error) / m
        db2 = np.sum(output_error, axis=0, keepdims=True) / m
        
        hidden_error = np.dot(output_error, self.W2.T) * self.activation_derivative(self.hidden_activation)
        dW1 = np.dot(X.T, hidden_error) / m
        db1 = np.sum(hidden_error, axis=0, keepdims=True) / m
        
        # Gradient descent update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # Store gradient magnitudes for visualization
        self.gradients = {'W1': np.abs(dW1), 'W2': np.abs(dW2)}

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int).reshape(-1, 1)  # Circular boundary
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform multiple training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Hidden Space Visualization (3D)
    hidden_features = mlp.hidden_activation
    ax_hidden.scatter(
        hidden_features[:, 0],
        hidden_features[:, 1],
        hidden_features[:, 2],
        c=y.ravel(),
        cmap='bwr',
        alpha=0.7
    )
    ax_hidden.set_title(f"Hidden Space at Step {frame}")
    ax_hidden.set_xlabel("h1")
    ax_hidden.set_ylabel("h2")
    ax_hidden.set_zlabel("h3")
    ax_hidden.view_init(30, 120)  # Rotate for better perspective

    # Decision Boundary in Input Space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.forward(grid).reshape(xx.shape)

    # Plot decision boundary
    ax_input.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap="bwr", alpha=0.6)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolors="k")
    ax_input.set_title(f"Input Space at Step {frame}")
    ax_input.set_xlim(x_min, x_max)
    ax_input.set_ylim(y_min, y_max)

    # Gradients Visualization with Edge Thickness
    node_labels = ["x1", "x2", "h1", "h2", "h3", "y"]
    node_positions = {
        "x1": (0.1, 0.8), "x2": (0.1, 0.2),
        "h1": (0.5, 0.7), "h2": (0.5, 0.5), "h3": (0.5, 0.3),
        "y": (0.9, 0.5),
    }

    grad_weights = np.concatenate([mlp.gradients["W1"].ravel(), mlp.gradients["W2"].ravel()])
    max_grad = np.max(grad_weights)
    edge_thickness = grad_weights / max_grad

    for i, (start, end) in enumerate([
        ("x1", "h1"), ("x1", "h2"), ("x1", "h3"),
        ("x2", "h1"), ("x2", "h2"), ("x2", "h3"),
        ("h1", "y"), ("h2", "y"), ("h3", "y"),
    ]):
        x_start, y_start = node_positions[start]
        x_end, y_end = node_positions[end]
        ax_gradient.plot(
            [x_start, x_end], [y_start, y_end], "k-", lw=2 * edge_thickness[i], alpha=0.7
        )

    for label, (x, y) in node_positions.items():
        ax_gradient.scatter(x, y, s=500, color="blue", alpha=0.8)
        ax_gradient.text(x, y, label, ha="center", va="center", color="white")
    ax_gradient.set_xlim(0, 1)
    ax_gradient.set_ylim(0, 1)
    ax_gradient.axis("off")
    ax_gradient.set_title(f"Gradients at Step {frame}")

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)