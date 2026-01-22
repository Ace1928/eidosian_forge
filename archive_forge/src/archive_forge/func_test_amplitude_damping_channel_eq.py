import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_amplitude_damping_channel_eq():
    a = cirq.amplitude_damp(0.0099999)
    b = cirq.amplitude_damp(0.01)
    c = cirq.amplitude_damp(0.0)
    assert cirq.approx_eq(a, b, atol=0.01)
    et = cirq.testing.EqualsTester()
    et.make_equality_group(lambda: c)
    et.add_equality_group(cirq.amplitude_damp(0.1))
    et.add_equality_group(cirq.amplitude_damp(0.4))
    et.add_equality_group(cirq.amplitude_damp(0.6))
    et.add_equality_group(cirq.amplitude_damp(0.8))