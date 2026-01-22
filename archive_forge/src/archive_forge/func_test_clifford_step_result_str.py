import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_clifford_step_result_str():
    q0 = cirq.LineQubit(0)
    result = next(cirq.CliffordSimulator().simulate_moment_steps(cirq.Circuit(cirq.measure(q0, key='m'))))
    assert str(result) == 'm=0\n|0⟩'