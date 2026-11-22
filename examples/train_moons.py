import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import make_moons

from micrograd.engine import Value
from micrograd.nn import MLP
from micrograd.visualize_graph import draw_dot


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RANDOM_SEED = 1337
NUM_SAMPLES = 100
NOISE = 0.1

EPOCHS = 100
INITIAL_LEARNING_RATE = 1.0
FINAL_LEARNING_RATE = 0.1
BATCH_SIZE = None
REGULARIZATION_ALPHA = 1e-4

OUTPUT_DIR = Path("outputs")

DATASET_PLOT_FILENAME = "two_moons_dataset.png"
GRAPH_FILENAME = "training_loss_graph"
DECISION_BOUNDARY_FILENAME = "decision_boundary.png"
GRAPH_FORMAT = "png"

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def create_dataset():
    X, y = make_moons(
        n_samples=NUM_SAMPLES,
        noise=NOISE,
        random_state=RANDOM_SEED
    )

    # Convert labels from 0/1 to -1/+1
    y = y * 2 - 1

    return X, y


def plot_dataset(X, y):
    plt.figure(figsize=(5, 5))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        s=20,
        cmap="jet"
    )

    plt.title("Two Moons Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    output_path = OUTPUT_DIR / DATASET_PLOT_FILENAME

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Dataset plot saved at:", output_path)


def create_model():
    model = MLP(2, [16, 16, 1])

    print(model)
    print("Number of parameters:", len(model.parameters()))

    return model


def calculate_loss(model, X, y, batch_size=None):

    if batch_size is None:
        X_batch, y_batch = X, y
    else:
        indices = np.random.permutation(X.shape[0])[:batch_size]
        X_batch, y_batch = X[indices], y[indices]

    inputs = [
        [Value(float(value)) for value in row]
        for row in X_batch
    ]

    # Forward pass
    scores = [model(x) for x in inputs]

    # SVM max-margin loss
    losses = [
        (1 + -label * score).relu()
        for label, score in zip(y_batch, scores)
    ]

    data_loss = sum(losses) / len(losses)

    # L2 regularization
    parameters = model.parameters()

    reg_loss = REGULARIZATION_ALPHA * sum(
        parameter * parameter
        for parameter in parameters
    )

    # Total loss
    total_loss = data_loss + reg_loss

    # Calculate accuracy
    accuracy = [
        (label > 0) == (score.data > 0)
        for label, score in zip(y_batch, scores)
    ]

    accuracy = sum(accuracy) / len(accuracy)

    return total_loss, accuracy


def train_model(model, X, y):

    for epoch in range(EPOCHS):
        # Forward pass
        total_loss, accuracy = calculate_loss(
            model, X, y, BATCH_SIZE
        )

        # Backward pass
        model.zero_grad()
        total_loss.backward()

        # Linear learning rate decay
        learning_rate = (
            INITIAL_LEARNING_RATE
            - (INITIAL_LEARNING_RATE - FINAL_LEARNING_RATE)
            * epoch / EPOCHS
        )

        # Update parameters using SGD
        for parameter in model.parameters():
            parameter.data -= learning_rate * parameter.grad

        # Print training progress
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {total_loss.data:.4f} | "
                f"Accuracy: {accuracy * 100:.2f}% | "
                f"LR: {learning_rate:.4f}"
            )


def visualize_computational_graph(model, X, y, sample_index=0):

    # Reset gradients
    model.zero_grad()

    # Select one training example
    x = [
        Value(float(value))
        for value in X[sample_index]
    ]

    label = y[sample_index]

    # Forward pass for one example
    score = model(x)

    # Calculate it's hinge loss
    sample_loss = (1 + -label * score).relu()

    # Backward pass
    sample_loss.backward()

    # Draw computational graph
    dot = draw_dot(sample_loss)

    # Save graph
    output_path = dot.render(
        filename=str(OUTPUT_DIR / GRAPH_FILENAME),
        format=GRAPH_FORMAT,
        cleanup=True
    )

    print("\nComputational graph saved at:", output_path)


def plot_decision_boundary(model, X, y):

    step_size = 0.25

    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1

    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step_size),
        np.arange(y_min, y_max, step_size)
    )

    # Create grid points
    X_mesh = np.c_[xx.ravel(), yy.ravel()]

    # Predict for every grid point
    inputs = [
        [Value(float(value)) for value in row]
        for row in X_mesh
    ]

    scores = [model(x) for x in inputs]

    predictions = np.array([
        score.data > 0
        for score in scores
    ])

    predictions = predictions.reshape(xx.shape)

    # Plot decision boundary and original data
    plt.figure(figsize=(7, 7))

    plt.contourf(
        xx, yy, predictions,
        cmap=plt.cm.Spectral,
        alpha=0.8
    )

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        s=40,
        cmap=plt.cm.Spectral
    )

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.title("Neural Network Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    output_path = OUTPUT_DIR / DECISION_BOUNDARY_FILENAME

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Decision boundary saved at:", output_path)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    X, y = create_dataset()

    # Visualize original dataset
    plot_dataset(X, y)

    # Create model
    model = create_model()

    # Check initial loss
    initial_loss, initial_accuracy = calculate_loss(model, X, y)

    print("\nInitial loss:", initial_loss.data)
    print("Initial accuracy:", initial_accuracy * 100, "%")

    # Train model
    print("\nStarting training...\n")
    train_model(model, X, y)

    # Visualize computational graph
    visualize_computational_graph(model, X, y)

    # Plot final decision boundary
    plot_decision_boundary(model, X, y)


if __name__ == "__main__":
    main()
