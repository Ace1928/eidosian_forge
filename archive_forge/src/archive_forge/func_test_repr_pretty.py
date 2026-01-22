import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('terms', ({}, {'Y': 2}, {'X': 1, 'Y': -1j}, {'X': np.sqrt(3) / 3, 'Y': np.sqrt(3) / 3, 'Z': np.sqrt(3) / 3}, {'I': np.sqrt(1j)}, {'X': np.sqrt(-1j)}, {cirq.X: 1, cirq.H: -1}))
def test_repr_pretty(terms):
    printer = FakePrinter()
    linear_dict = cirq.LinearDict(terms)
    linear_dict._repr_pretty_(printer, False)
    assert printer.buffer.replace(' ', '') == str(linear_dict).replace(' ', '')
    printer.reset()
    linear_dict._repr_pretty_(printer, True)
    assert printer.buffer == 'LinearDict(...)'