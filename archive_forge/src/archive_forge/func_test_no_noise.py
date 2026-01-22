from typing import Sequence
import numpy as np
import pytest
import cirq
from cirq import ops
from cirq.devices.noise_model import validate_all_measurements
from cirq.testing import assert_equivalent_op_tree
def test_no_noise():
    q = cirq.LineQubit(0)
    m = cirq.Moment([cirq.X(q)])
    assert cirq.NO_NOISE.noisy_operation(cirq.X(q)) == cirq.X(q)
    assert cirq.NO_NOISE.noisy_moment(m, [q]) is m
    assert cirq.NO_NOISE.noisy_moments([m, m], [q]) == [m, m]
    assert cirq.NO_NOISE == cirq.NO_NOISE
    assert str(cirq.NO_NOISE) == '(no noise)'
    cirq.testing.assert_equivalent_repr(cirq.NO_NOISE)