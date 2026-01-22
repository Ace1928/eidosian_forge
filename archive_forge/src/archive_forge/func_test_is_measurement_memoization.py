import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_is_measurement_memoization():
    a = cirq.LineQubit(0)
    circuit = cirq.FrozenCircuit(cirq.measure(a, key='m'))
    c_op = cirq.CircuitOperation(circuit)
    cache_name = _compat._method_cache_name(circuit._is_measurement_)
    assert not hasattr(circuit, cache_name)
    assert cirq.is_measurement(c_op)
    assert hasattr(circuit, cache_name)