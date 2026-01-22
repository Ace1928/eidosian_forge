import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_compute_amplitudes_bad_input():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.X(a), cirq.H(a), cirq.H(b))
    sim = cirq.Simulator()
    with pytest.raises(ValueError, match='1-dimensional'):
        _ = sim.compute_amplitudes(c, np.array([[0, 0]]))