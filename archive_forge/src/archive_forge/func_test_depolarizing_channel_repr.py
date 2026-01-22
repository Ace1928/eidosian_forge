import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_depolarizing_channel_repr():
    cirq.testing.assert_equivalent_repr(cirq.DepolarizingChannel(0.3))