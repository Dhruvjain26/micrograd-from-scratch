import pytest
from micrograd.engine import Value
from micrograd.nn import Module, Neuron, Layer, MLP


# MODULE

def test_module_parameters():
    module = Module()

    assert module.parameters() == []

def test_module_zero_grad():
    module = Module()
    module.zero_grad()

    assert module.parameters() == []


# NEURON

def test_neuron_forward():
    neuron = Neuron(2, nonLin=False)
    neuron.weights[0].data = 2.0
    neuron.weights[1].data = 3.0
    neuron.bias.data = 1.0
    x = [Value(4.0), Value(5.0)]
    output = neuron(x)

    # (2 * 4) + (3 * 5) + 1 = 24
    assert output.data == 24.0

def test_neuron_relu():
    neuron = Neuron(2, nonLin=True)
    neuron.weights[0].data = -2.0
    neuron.weights[1].data = -3.0
    neuron.bias.data = 0.0
    x = [Value(4.0), Value(5.0)]
    output = neuron(x)

    assert output.data == 0.0

def test_neuron_parameters():
    neuron = Neuron(3)
    params = neuron.parameters()

    # 3 weights + 1 bias
    assert len(params) == 4
    assert neuron.weights + [neuron.bias] == params


# LAYER

def test_layer_forward():
    layer = Layer(2, 3, nonLin=False)
    x = [Value(2.0), Value(3.0)]
    output = layer(x)

    assert len(output) == 3
    assert all(isinstance(value, Value) for value in output)

def test_layer_single_neuron_output():
    layer = Layer(2, 1)
    x = [Value(2.0), Value(3.0)]
    output = layer(x)

    assert isinstance(output, Value)

def test_layer_parameters():
    layer = Layer(2, 3)
    params = layer.parameters()

    # 3 neurons * (2 weights + 1 bias)
    assert len(params) == 9


# MLP

def test_mlp_forward():
    model = MLP(2, [3, 1])
    x = [Value(2.0), Value(3.0)]
    output = model(x)

    assert isinstance(output, Value)

def test_mlp_parameters():
    model = MLP(2, [3, 1])

    # 3 neurons * (2 weights + 1 bias) = 9 + 1 neuron * (3 weights + 1 bias) = 4
    assert len(model.parameters()) == 13


# BACKPROPAGATION THROUGH MLP

def test_mlp_backward():
    model = MLP(2, [2, 1])
    params = model.parameters()
    values = [
        1.0, 1.0, 1.0,   # First neuron of hidden layer: weights, bias
        1.0, -1.0, 2.0,  # Second neuron of hidden layer: weights, bias
        1.0, 1.0, 0.0    # Output neuron: weights, bias
    ]
    for param, value in zip(params, values):
        param.data = value
    x = [Value(1.0), Value(2.0)]
    output = model(x)

    # Hidden layer:
    # Neuron 1: 1*1 + 1*2 + 1 = 4
    # Neuron 2: 1*1 + (-1)*2 + 2 = 1
    #
    # Output layer:
    # 4*1 + 1*1 + 0 = 5
    assert output.data == 5.0

    output.backward()

    # All parameters received gradients.
    assert all(p.grad is not None for p in params)
    # Check that at least one gradient is non-zero.
    assert any(p.grad != 0 for p in params)


# ZERO GRADIENTS

def test_mlp_zero_grad():
    model = MLP(2, [3, 1])
    x = [Value(2.0), Value(3.0)]
    output = model(x)
    output.backward()

    # Gradients exist before resetting.
    assert any(p.grad != 0 for p in model.parameters())

    model.zero_grad()

    # All parameter gradients are now zero.
    assert all(p.grad == 0 for p in model.parameters())
