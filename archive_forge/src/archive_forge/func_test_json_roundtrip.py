import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_json_roundtrip():
    q0, q1, q2 = (cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(2))
    state = cirq.CliffordState(qubit_map={q0: 0, q1: 1, q2: 2})
    state.apply_unitary(cirq.X(q0))
    state.apply_unitary(cirq.H(q1))
    with pytest.raises(ValueError, match='T cannot be run with Clifford simulator.'):
        state.apply_unitary(cirq.T(q1))
    state_roundtrip = cirq.CliffordState._from_json_dict_(**state._json_dict_())
    state.apply_unitary(cirq.S(q1))
    state_roundtrip.apply_unitary(cirq.S(q1))
    assert np.allclose(state.ch_form.state_vector(), state_roundtrip.ch_form.state_vector())