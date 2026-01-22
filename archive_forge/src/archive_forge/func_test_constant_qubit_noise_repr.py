from typing import Sequence
import numpy as np
import pytest
import cirq
from cirq import ops
from cirq.devices.noise_model import validate_all_measurements
from cirq.testing import assert_equivalent_op_tree
def test_constant_qubit_noise_repr():
    cirq.testing.assert_equivalent_repr(cirq.ConstantQubitNoiseModel(cirq.X ** 0.01))