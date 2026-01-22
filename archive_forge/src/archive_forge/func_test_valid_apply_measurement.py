import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_valid_apply_measurement():
    q0 = cirq.LineQubit(0)
    state = cirq.CliffordState(qubit_map={q0: 0}, initial_state=1)
    measurements = {}
    state.apply_measurement(cirq.measure(q0), measurements, np.random.RandomState())
    assert measurements == {'q(0)': [1]}