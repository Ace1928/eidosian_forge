import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
@pytest.mark.parametrize('sim', ALL_SIMULATORS)
def test_repeat_until_sympy(sim):
    q1, q2 = cirq.LineQubit.range(2)
    circuitop = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q2), cirq.measure(q2, key='b')), use_repetition_ids=False, repeat_until=cirq.SympyCondition(sympy.Eq(sympy.Symbol('a'), sympy.Symbol('b'))))
    c = cirq.Circuit(cirq.measure(q1, key='a'), circuitop)
    assert len(c) == 2
    assert cirq.control_keys(circuitop) == {cirq.MeasurementKey('a')}
    measurements = sim.run(c).records['b'][0]
    assert len(measurements) == 2
    assert measurements[0] == (1,)
    assert measurements[1] == (0,)