import cirq
import numpy as np
import pytest
from cirq_ft.infra.decompose_protocol import (
def test_known_decomposition_empty_unitary():

    class DecomposeEmptyList(cirq.testing.SingleQubitGate):

        def _decompose_(self, _):
            return []
    gate = DecomposeEmptyList()
    assert _decompose_once_considering_known_decomposition(gate) == []