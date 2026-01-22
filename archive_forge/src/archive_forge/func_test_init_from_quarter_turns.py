import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_init_from_quarter_turns():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 0), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Y, 0), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Z, 0), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 4), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Y, 4), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Z, 4), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 8), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Y, 8), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Z, 8), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, -4), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Y, -4), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Z, -4))
    eq.add_equality_group(cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 1), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 5), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 9), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, -3))
    eq.add_equality_group(cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Y, 1), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Y, 5), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Y, 9), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Y, -3))
    eq.add_equality_group(cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Z, 1), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Z, 5), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Z, 9), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Z, -3))
    eq.add_equality_group(cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 2), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 6))
    eq.add_equality_group(cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 3), cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 7))