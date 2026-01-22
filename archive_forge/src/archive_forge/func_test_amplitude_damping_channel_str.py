import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_amplitude_damping_channel_str():
    assert str(cirq.amplitude_damp(0.3)) == 'amplitude_damp(gamma=0.3)'