import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_density_matrix_copy():
    sim = cirq.DensityMatrixSimulator(split_untangled_states=False)
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.H(q))
    matrices = []
    for step in sim.simulate_moment_steps(circuit):
        matrices.append(step.density_matrix(copy=True))
    assert all((np.isclose(np.trace(x), 1.0) for x in matrices))
    for x, y in itertools.combinations(matrices, 2):
        assert not np.shares_memory(x, y)
    matrices = []
    traces = []
    for step in sim.simulate_moment_steps(circuit):
        matrices.append(step.density_matrix(copy=False))
        traces.append(np.trace(step.density_matrix(copy=False)))
    assert any((not np.isclose(np.trace(x), 1.0) for x in matrices))
    assert all((np.isclose(x, 1.0) for x in traces))
    assert all((not np.shares_memory(x, y) for x, y in itertools.combinations(matrices, 2)))