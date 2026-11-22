import torch
import pytest
from micrograd.engine import Value

# Tests for basic operations and backward propagation in the Value class.

def test_value_initialization():
    a = Value(5.0)

    assert a.data == 5.0
    assert a.grad == 0.0

def test_addition_forward():
    a = Value(2.0)
    b = Value(3.0)

    c = a + b

    assert c.data == 5.0

def test_addition_backward():
    a = Value(2.0)
    b = Value(3.0)

    c = a + b
    c.backward()

    assert a.grad == 1.0
    assert b.grad == 1.0

def test_subtraction_forward():
    a = Value(5.0)
    b = Value(3.0)

    c = a - b

    assert c.data == 2.0

def test_subtraction_backward():
    a = Value(5.0)
    b = Value(3.0)

    c = a - b
    c.backward()

    assert a.grad == 1.0
    assert b.grad == -1.0

def test_multiplication_forward():
    a = Value(2.0)
    b = Value(3.0)

    c = a * b

    assert c.data == 6.0

def test_multiplication_backward():
    a = Value(2.0)
    b = Value(3.0)

    c = a * b
    c.backward()

    assert a.grad == 3.0
    assert b.grad == 2.0

def test_division_forward():
    a = Value(6.0)
    b = Value(3.0)

    c = a / b

    assert c.data == 2.0

def test_division_backward():
    a = Value(6.0)
    b = Value(3.0)

    c = a / b
    c.backward()

    assert a.grad == pytest.approx(1 / 3)
    assert b.grad == pytest.approx(-2 / 3)

def test_power_forward():
    a = Value(3.0)

    c = a ** 2

    assert c.data == 9.0

def test_power_backward():
    a = Value(3.0)

    c = a ** 2
    c.backward()

    assert a.grad == 6.0

def test_relu_forward_positive():
    a = Value(5.0)

    c = a.relu()

    assert c.data == 5.0

def test_relu_forward_negative():
    a = Value(-5.0)

    c = a.relu()

    assert c.data == 0.0

def test_relu_backward_positive():
    a = Value(5.0)

    c = a.relu()
    c.backward()

    assert a.grad == 1.0

def test_relu_backward_negative():
    a = Value(-5.0)

    c = a.relu()
    c.backward()

    assert a.grad == 0.0

def test_gradient_accumulation():
    a = Value(3.0)

    c = a * a

    c.backward()

    assert a.grad == 6.0

def test_complex_expression():
    a = Value(2.0)
    b = Value(3.0)

    c = a * b + a

    c.backward()

    assert c.data == 8.0
    assert a.grad == 4.0
    assert b.grad == 2.0

# Testing Value class vs pytorch

def test_value_vs_pytorch_simple_ops():
    x = Value(-4.0)

    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()

    xvc, yvc = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()

    xpt, ypt = x, y

    tol = 1e-6
    # forward pass
    assert abs(yvc.data - ypt.data.item()) < tol
    # backward pass
    assert abs(xvc.grad - xpt.grad.item()) < tol

def test_value_vs_pytorch_more_ops():
    a = Value(-2.0)
    b = Value(3.0)

    c = a + b
    d = a * b + b ** 3
    c = c + 1
    d = d + (c + a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + (10.0 / f)
    g.backward()

    avc, bvc, gvc = a, b, g

    a = torch.Tensor([-2.0]).double()
    a.requires_grad = True
    b = torch.Tensor([3.0]).double()
    b.requires_grad = True

    c = a + b
    d = a * b + b ** 3
    c = c + 1
    d = d + (c + a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + (10.0 / f)
    g.backward()

    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass
    assert abs(gvc.data - gpt.data.item()) < tol
    # backward pass
    assert abs(avc.grad - apt.grad.item()) < tol
    assert abs(bvc.grad - bpt.grad.item()) < tol