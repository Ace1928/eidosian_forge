import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_raises_for_non_commuting_paulis():
    with pytest.raises(ValueError, match='commuting'):
        cirq.PauliSumExponential(cirq.X(q0) + cirq.Z(q0), np.pi / 2)