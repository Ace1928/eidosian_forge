import numpy as np
import pytest
import cirq
from cirq.devices.noise_utils import (
def test_op_id_swap():
    q0, q1 = cirq.LineQubit.range(2)
    base_id = OpIdentifier(cirq.CZPowGate, q0, q1)
    swap_id = OpIdentifier(base_id.gate_type, *base_id.qubits[::-1])
    assert cirq.CZ(q0, q1) in base_id
    assert cirq.CZ(q0, q1) not in swap_id
    assert cirq.CZ(q1, q0) not in base_id
    assert cirq.CZ(q1, q0) in swap_id