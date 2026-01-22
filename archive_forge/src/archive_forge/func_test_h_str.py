import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_h_str():
    assert str(cirq.H) == 'H'
    assert str(cirq.H ** 0.5) == 'H**0.5'