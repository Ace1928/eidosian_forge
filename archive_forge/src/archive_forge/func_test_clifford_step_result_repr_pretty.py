import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_clifford_step_result_repr_pretty():
    q0 = cirq.LineQubit(0)
    result = next(cirq.CliffordSimulator().simulate_moment_steps(cirq.Circuit(cirq.measure(q0, key='m'))))
    cirq.testing.assert_repr_pretty(result, 'm=0\n|0⟩')
    cirq.testing.assert_repr_pretty(result, 'cirq.CliffordSimulatorStateResult(...)', cycle=True)