import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_cz_init():
    assert cirq.CZPowGate(exponent=0.5).exponent == 0.5
    assert cirq.CZPowGate(exponent=5).exponent == 5
    assert (cirq.CZ ** 0.5).exponent == 0.5