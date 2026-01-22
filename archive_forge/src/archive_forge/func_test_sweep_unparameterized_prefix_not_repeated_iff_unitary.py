import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
def test_sweep_unparameterized_prefix_not_repeated_iff_unitary():
    q = cirq.LineQubit(0)

    class TestOp(cirq.Operation):

        def __init__(self, *, has_unitary: bool):
            self.count = 0
            self.has_unitary = has_unitary

        def _act_on_(self, sim_state):
            self.count += 1
            return True

        def with_qubits(self, qubits):
            pass

        @property
        def qubits(self):
            return (q,)

        def _has_unitary_(self):
            return self.has_unitary
    simulator = CountingSimulator()
    params = [cirq.ParamResolver({'a': 0}), cirq.ParamResolver({'a': 1})]
    op1 = TestOp(has_unitary=True)
    op2 = TestOp(has_unitary=True)
    circuit = cirq.Circuit(op1, cirq.XPowGate(exponent=sympy.Symbol('a'))(q), op2)
    rs = simulator.simulate_sweep(program=circuit, params=params)
    assert rs[0]._final_simulator_state.copy_count == 1
    assert rs[1]._final_simulator_state.copy_count == 0
    assert op1.count == 1
    assert op2.count == 2
    op1 = TestOp(has_unitary=False)
    op2 = TestOp(has_unitary=False)
    circuit = cirq.Circuit(op1, cirq.XPowGate(exponent=sympy.Symbol('a'))(q), op2)
    rs = simulator.simulate_sweep(program=circuit, params=params)
    assert rs[0]._final_simulator_state.copy_count == 1
    assert rs[1]._final_simulator_state.copy_count == 0
    assert op1.count == 2
    assert op2.count == 2