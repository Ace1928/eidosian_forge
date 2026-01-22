import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_keys_under_parent_path():
    a = cirq.LineQubit(0)
    op1 = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(a, key='A')))
    assert cirq.measurement_key_names(op1) == {'A'}
    op2 = op1.with_key_path(('B',))
    assert cirq.measurement_key_names(op2) == {'B:A'}
    op3 = cirq.with_key_path_prefix(op2, ('C',))
    assert cirq.measurement_key_names(op3) == {'C:B:A'}
    op4 = op3.repeat(2)
    assert cirq.measurement_key_names(op4) == {'C:B:0:A', 'C:B:1:A'}