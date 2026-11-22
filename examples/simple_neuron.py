from pathlib import Path

from micrograd.engine import Value
from micrograd.nn import Neuron
from micrograd.visualize_graph import draw_dot


OUTPUT_DIR = Path("outputs")
GRAPH_FILENAME = "simple_neuron_graph"
GRAPH_FORMAT = "png"


def main():

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create a neuron with 2 inputs
    neuron = Neuron(2)

    # Input values
    x = [Value(2.0), Value(3.0)]

    # Forward pass
    output = neuron(x)

    print("Neuron output:", output.data)

    # Backward pass
    output.backward()

    # Display input gradients
    print("\nInput gradients:")
    for i, xi in enumerate(x):
        print(f"Input {i + 1}: {xi.grad}")

    # Display parameter gradients
    print("\nNeuron parameter gradients:")
    for p in neuron.parameters():
        print(p.grad)

    # Visualize computational graph
    dot = draw_dot(output)

    # Save computational graph
    output_path = dot.render(
        filename=str(OUTPUT_DIR / GRAPH_FILENAME),
        format=GRAPH_FORMAT,
        cleanup=True
    )

    print("\nComputational graph saved at:", output_path)


if __name__ == "__main__":
    main()
