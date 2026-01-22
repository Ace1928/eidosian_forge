import numpy as np
import pytest
import sympy
import cirq
def test_ms_str():
    ms = cirq.ms(np.pi / 2)
    assert str(ms) == 'MS(π/2)'
    assert str(cirq.ms(np.pi)) == 'MS(2.0π/2)'
    assert str(ms ** 0.5) == 'MS(0.5π/2)'
    assert str(ms ** 2) == 'MS(2.0π/2)'
    assert str(ms ** (-1)) == 'MS(-1.0π/2)'