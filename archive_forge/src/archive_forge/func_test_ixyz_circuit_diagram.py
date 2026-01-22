import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_ixyz_circuit_diagram():
    q = cirq.NamedQubit('q')
    ix = cirq.XPowGate(exponent=1, global_shift=0.5)
    iy = cirq.YPowGate(exponent=1, global_shift=0.5)
    iz = cirq.ZPowGate(exponent=1, global_shift=0.5)
    cirq.testing.assert_has_diagram(cirq.Circuit(ix(q), ix(q) ** (-1), ix(q) ** (-0.99999), ix(q) ** (-1.00001), ix(q) ** 3, ix(q) ** 4.5, ix(q) ** 4.500001), '\nq: ───X───X───X───X───X───X^0.5───X^0.5───\n        ')
    cirq.testing.assert_has_diagram(cirq.Circuit(iy(q), iy(q) ** (-1), iy(q) ** 3, iy(q) ** 4.5, iy(q) ** 4.500001), '\nq: ───Y───Y───Y───Y^0.5───Y^0.5───\n    ')
    cirq.testing.assert_has_diagram(cirq.Circuit(iz(q), iz(q) ** (-1), iz(q) ** 3, iz(q) ** 4.5, iz(q) ** 4.500001), '\nq: ───Z───Z───Z───S───S───\n    ')