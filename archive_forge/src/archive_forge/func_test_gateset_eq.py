from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
def test_gateset_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.Gateset(CustomX))
    eq.add_equality_group(cirq.Gateset(CustomX ** 3))
    eq.add_equality_group(cirq.Gateset(CustomX, name='Custom Gateset'))
    eq.add_equality_group(cirq.Gateset(CustomX, name='Custom Gateset', unroll_circuit_op=False))
    eq.add_equality_group(cirq.Gateset(CustomX, cirq.GlobalPhaseGate, name='Custom Gateset'))
    eq.add_equality_group(cirq.Gateset(cirq.GateFamily(CustomX, name='custom_name', description='custom_description'), cirq.GateFamily(CustomX, name='custom_name', description='custom_description')), cirq.Gateset(cirq.GateFamily(CustomX ** 3, name='custom_name', description='custom_description'), cirq.GateFamily(CustomX, name='custom_name', description='custom_description')))
    eq.add_equality_group(cirq.Gateset(CustomX, CustomXPowGate), cirq.Gateset(CustomXPowGate, CustomX), cirq.Gateset(CustomX, CustomX, CustomXPowGate), cirq.Gateset(CustomXPowGate, CustomX, CustomXPowGate))
    eq.add_equality_group(cirq.Gateset(CustomXGateFamily()))
    eq.add_equality_group(cirq.Gateset(cirq.GateFamily(gate=CustomXPowGate, name='CustomXGateFamily', description='Accepts all integer powers of CustomXPowGate')))