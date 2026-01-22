from typing import Union, Tuple, cast
import numpy as np
import pytest
import sympy
import cirq
from cirq.type_workarounds import NotImplementedType
def test_pow_inverse():
    assert cirq.inverse(CRestricted, None) is None
    assert cirq.pow(CRestricted, 1.5, None) is None
    assert cirq.pow(CY, 1.5) == cirq.ControlledGate(cirq.Y ** 1.5)
    assert cirq.inverse(CY) == CY ** (-1) == CY
    assert cirq.inverse(C0Restricted, None) is None
    assert cirq.pow(C0Restricted, 1.5, None) is None
    assert cirq.pow(C0Y, 1.5) == cirq.ControlledGate(cirq.Y ** 1.5, control_values=[0])
    assert cirq.inverse(C0Y) == C0Y ** (-1) == C0Y
    assert cirq.inverse(C2Restricted, None) is None
    assert cirq.pow(C2Restricted, 1.5, None) is None
    assert cirq.pow(C2Y, 1.5) == cirq.ControlledGate(cirq.Y ** 1.5, control_values=[2], control_qid_shape=(3,))
    assert cirq.inverse(C2Y) == C2Y ** (-1) == C2Y