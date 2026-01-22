import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_mapped_circuit_keeps_keys_under_parent_path():
    q = cirq.LineQubit(0)
    op1 = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(q, key='A'), cirq.measure_single_paulistring(cirq.X(q), key='B'), cirq.MixedUnitaryChannel.from_mixture(cirq.bit_flip(0.5), key='C').on(q), cirq.KrausChannel.from_channel(cirq.phase_damp(0.5), key='D').on(q)))
    op2 = op1.with_key_path(('X',))
    assert cirq.measurement_key_names(op2.mapped_circuit()) == {'X:A', 'X:B', 'X:C', 'X:D'}