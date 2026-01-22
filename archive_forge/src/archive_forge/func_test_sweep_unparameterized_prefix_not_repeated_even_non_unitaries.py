import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_sweep_unparameterized_prefix_not_repeated_even_non_unitaries():
    q = cirq.LineQubit(0)

    class NonUnitaryOp(cirq.Operation):
        count = 0

        def _act_on_(self, sim_state):
            self.count += 1
            return True

        def with_qubits(self, qubits):
            pass

        @property
        def qubits(self):
            return (q,)
    simulator = cirq.DensityMatrixSimulator()
    params = [cirq.ParamResolver({'a': 0}), cirq.ParamResolver({'a': 1})]
    op1 = NonUnitaryOp()
    op2 = NonUnitaryOp()
    circuit = cirq.Circuit(op1, cirq.XPowGate(exponent=sympy.Symbol('a'))(q), op2)
    simulator.simulate_sweep(program=circuit, params=params)
    assert op1.count == 1
    assert op2.count == 2