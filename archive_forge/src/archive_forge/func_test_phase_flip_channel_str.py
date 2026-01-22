import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_phase_flip_channel_str():
    assert str(cirq.phase_flip(0.3)) == 'phase_flip(p=0.3)'