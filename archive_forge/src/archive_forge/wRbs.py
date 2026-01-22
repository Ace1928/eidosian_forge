# calculator.py
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
# test_calculator.py
import pytest
from calculator import add, subtract
def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(-1, 1) == -2
    assert subtract(0, 0) == 0
def test_add_invalid_input():
    with pytest.raises(TypeError):
        add('2', 3)
def test_subtract_invalid_input():
    with