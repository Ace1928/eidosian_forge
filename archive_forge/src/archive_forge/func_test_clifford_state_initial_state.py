import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_clifford_state_initial_state():
    q0 = cirq.LineQubit(0)
    with pytest.raises(ValueError, match='Out of range'):
        _ = cirq.CliffordState(qubit_map={q0: 0}, initial_state=2)
    state = cirq.CliffordState(qubit_map={q0: 0}, initial_state=1)
    np.testing.assert_allclose(state.state_vector(), [0, 1])