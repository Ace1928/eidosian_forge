import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_works_on_pauli_string_phasor():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(np.exp(0.5j * np.pi * cirq.X(a) * cirq.X(b)))
    sim = cirq.DensityMatrixSimulator()
    result = sim.simulate(c).final_density_matrix
    np.testing.assert_allclose(result.reshape(4, 4), np.diag([0, 0, 0, 1]), atol=1e-08)