import random
import numpy as np
import pytest
import sympy
import cirq
def test_str_diagram():
    g = cirq.PhasedXZGate(x_exponent=0.5, z_exponent=0.25, axis_phase_exponent=0.125)
    assert str(g) == 'PhXZ(a=0.125,x=0.5,z=0.25)'
    cirq.testing.assert_has_diagram(cirq.Circuit(g.on(cirq.LineQubit(0))), '\n0: ───PhXZ(a=0.125,x=0.5,z=0.25)───\n    ')