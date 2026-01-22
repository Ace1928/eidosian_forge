import itertools
from typing import Any
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_identity_pow():
    I = cirq.I
    q = cirq.NamedQubit('q')
    assert I(q) ** 0.5 == I(q)
    assert I(q) ** 2 == I(q)
    assert I(q) ** (1 + 1j) == I(q)
    assert I(q) ** sympy.Symbol('x') == I(q)
    with pytest.raises(TypeError):
        _ = (I ** q)(q)
    with pytest.raises(TypeError):
        _ = I(q) ** q