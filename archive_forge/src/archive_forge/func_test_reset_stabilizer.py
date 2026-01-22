import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_reset_stabilizer():
    assert cirq.has_stabilizer_effect(cirq.reset(cirq.LineQubit(0)))