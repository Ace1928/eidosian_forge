from typing import Union, Tuple, cast
import numpy as np
import pytest
import sympy
import cirq
from cirq.type_workarounds import NotImplementedType
def test_reversible():
    assert cirq.inverse(cirq.ControlledGate(cirq.S)) == cirq.ControlledGate(cirq.S ** (-1))
    assert cirq.inverse(cirq.ControlledGate(cirq.S, num_controls=4)) == cirq.ControlledGate(cirq.S ** (-1), num_controls=4)
    assert cirq.inverse(cirq.ControlledGate(cirq.S, control_values=[1])) == cirq.ControlledGate(cirq.S ** (-1), control_values=[1])
    assert cirq.inverse(cirq.ControlledGate(cirq.S, control_qid_shape=(3,))) == cirq.ControlledGate(cirq.S ** (-1), control_qid_shape=(3,))