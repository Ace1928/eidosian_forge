import numpy as np
import pytest
import cirq
def test_str_full():
    t = cirq.CliffordTableau(num_qubits=1)
    expected_str = 'stable | destable\n-------+----------\n+ Z0   | + X0\n'
    assert t._str_full_() == expected_str
    t = cirq.CliffordTableau(num_qubits=1)
    _S(t, 0)
    expected_str = 'stable | destable\n-------+----------\n+ Z0   | + Y0\n'
    assert t._str_full_() == expected_str
    t = cirq.CliffordTableau(num_qubits=2)
    expected_str = 'stable | destable\n-------+----------\n+ Z0   | + X0  \n+   Z1 | +   X1\n'
    assert t._str_full_() == expected_str