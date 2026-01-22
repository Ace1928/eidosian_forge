import itertools
import re
from typing import cast, Tuple, Union
import numpy as np
import pytest
import sympy
import cirq
from cirq import protocols
from cirq.type_workarounds import NotImplementedType
def test_controlled_circuit_operation_is_consistent():
    op = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.XXPowGate(exponent=0.25, global_shift=-0.5).on(*cirq.LineQubit.range(2))))
    cb = cirq.NamedQubit('ctr')
    cop = cirq.ControlledOperation([cb], op)
    cirq.testing.assert_implements_consistent_protocols(cop, exponents=(-1, 1, 2))
    cirq.testing.assert_decompose_ends_at_default_gateset(cop)
    cop = cirq.ControlledOperation([cb], op, control_values=[0])
    cirq.testing.assert_implements_consistent_protocols(cop, exponents=(-1, 1, 2))
    cirq.testing.assert_decompose_ends_at_default_gateset(cop)
    cop = cirq.ControlledOperation([cb], op, control_values=[(0, 1)])
    cirq.testing.assert_implements_consistent_protocols(cop, exponents=(-1, 1, 2))
    cirq.testing.assert_decompose_ends_at_default_gateset(cop)