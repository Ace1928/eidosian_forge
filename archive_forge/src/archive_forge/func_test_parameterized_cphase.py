import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_parameterized_cphase():
    assert cirq.cphase(sympy.pi) == cirq.CZ
    assert cirq.cphase(sympy.pi / 2) == cirq.CZ ** 0.5