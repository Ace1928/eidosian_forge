import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_h_init():
    h = cirq.HPowGate(exponent=0.5)
    assert h.exponent == 0.5