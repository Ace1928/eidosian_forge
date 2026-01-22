import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_phase_damp_consistency():
    full_damp = cirq.PhaseDampingChannel(gamma=1)
    cirq.testing.assert_has_consistent_apply_channel(full_damp)
    partial_damp = cirq.PhaseDampingChannel(gamma=0.5)
    cirq.testing.assert_has_consistent_apply_channel(partial_damp)
    no_damp = cirq.PhaseDampingChannel(gamma=0)
    cirq.testing.assert_has_consistent_apply_channel(no_damp)