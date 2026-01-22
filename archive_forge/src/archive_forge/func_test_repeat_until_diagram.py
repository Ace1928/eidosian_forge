import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_repeat_until_diagram():
    q = cirq.LineQubit(0)
    key = cirq.MeasurementKey('m')
    c = cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q) ** 0.2, cirq.measure(q, key=key)), use_repetition_ids=False, repeat_until=cirq.KeyCondition(key)))
    cirq.testing.assert_has_diagram(c, "\n0: ───[ 0: ───X^0.2───M('m')─── ](no_rep_ids, until=m)───\n", use_unicode_characters=True)