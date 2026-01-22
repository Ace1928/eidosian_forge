import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_generalized_amplitude_damping_str():
    assert str(cirq.generalized_amplitude_damp(0.1, 0.3)) == 'generalized_amplitude_damp(p=0.1,gamma=0.3)'