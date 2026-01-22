import random
import numpy as np
import pytest
import sympy
import cirq
def test_init_properties():
    g = cirq.PhasedXZGate(x_exponent=0.125, z_exponent=0.25, axis_phase_exponent=0.375)
    assert g.x_exponent == 0.125
    assert g.z_exponent == 0.25
    assert g.axis_phase_exponent == 0.375