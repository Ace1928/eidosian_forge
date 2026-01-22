import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_repr_operation():
    cirq.testing.assert_equivalent_repr(cirq.SingleQubitCliffordGate.from_pauli(cirq.Z).on(cirq.LineQubit(2)))