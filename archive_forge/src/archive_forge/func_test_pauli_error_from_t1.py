import numpy as np
import pytest
import cirq
from cirq.devices.noise_utils import (
@pytest.mark.parametrize('t,t1_ns,expected_output', [(20, 100000.0, (1 - np.exp(-20 / 200000.0)) / 2 + (1 - np.exp(-20 / 100000.0)) / 4), (4000, 10000.0, (1 - np.exp(-4000 / 20000.0)) / 2 + (1 - np.exp(-4000 / 10000.0)) / 4)])
def test_pauli_error_from_t1(t, t1_ns, expected_output):
    val = pauli_error_from_t1(t, t1_ns)
    assert val == expected_output