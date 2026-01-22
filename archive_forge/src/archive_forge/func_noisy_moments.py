from typing import Sequence
import numpy as np
import pytest
import cirq
from cirq import ops
from cirq.devices.noise_model import validate_all_measurements
from cirq.testing import assert_equivalent_op_tree
def noisy_moments(self, moments, system_qubits):
    result = []
    for moment in moments:
        if moment.operations:
            result.append(cirq.X(moment.operations[0].qubits[0]).with_tags(ops.VirtualTag()))
        else:
            result.append([])
    return result