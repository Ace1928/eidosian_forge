import cirq
import numpy as np
import pytest
def test_mix_mismatch_fails():
    op2 = np.zeros((4, 4))
    op2[1][1] = 1
    mix = [(0.5, np.array([[1, 0], [0, 0]])), (0.5, op2)]
    with pytest.raises(ValueError, match='Inconsistent unitary shapes'):
        _ = cirq.MixedUnitaryChannel(mixture=mix, key='m')