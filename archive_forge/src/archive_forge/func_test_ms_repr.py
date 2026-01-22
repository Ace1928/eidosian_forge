import numpy as np
import pytest
import sympy
import cirq
def test_ms_repr():
    assert repr(cirq.ms(np.pi / 2)) == 'cirq.ms(np.pi/2)'
    assert repr(cirq.ms(np.pi / 4)) == 'cirq.ms(0.5*np.pi/2)'
    cirq.testing.assert_equivalent_repr(cirq.ms(np.pi / 4))
    ms = cirq.ms(np.pi / 2)
    assert repr(ms ** 2) == 'cirq.ms(2.0*np.pi/2)'
    assert repr(ms ** (-0.5)) == 'cirq.ms(-0.5*np.pi/2)'