import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
@pytest.mark.parametrize('theta,pi', [(0.4, np.pi), (sympy.Symbol('theta'), sympy.pi)])
def test_transformations(theta, pi):
    initialRx = cirq.rx(theta)
    expectedPowx = cirq.X ** (theta / pi)
    receivedPowx = initialRx.with_canonical_global_phase()
    backToRx = receivedPowx.in_su2()
    assert receivedPowx == expectedPowx
    assert backToRx == initialRx
    initialRy = cirq.ry(theta)
    expectedPowy = cirq.Y ** (theta / pi)
    receivedPowy = initialRy.with_canonical_global_phase()
    backToRy = receivedPowy.in_su2()
    assert receivedPowy == expectedPowy
    assert backToRy == initialRy
    initialRz = cirq.rz(theta)
    expectedPowz = cirq.Z ** (theta / pi)
    receivedPowz = initialRz.with_canonical_global_phase()
    backToRz = receivedPowz.in_su2()
    assert receivedPowz == expectedPowz
    assert backToRz == initialRz