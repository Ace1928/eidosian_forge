import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_invalid_apply_measurement():
    q0 = cirq.LineQubit(0)
    state = cirq.CliffordState(qubit_map={q0: 0})
    measurements = {}
    with pytest.raises(TypeError, match='only supports cirq.MeasurementGate'):
        state.apply_measurement(cirq.H(q0), measurements, np.random.RandomState())
    assert measurements == {}