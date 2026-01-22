import cirq
import numpy as np
import pytest
def test_ops_mismatch_fails():
    op2 = np.zeros((4, 4))
    op2[1][1] = 1
    ops = [np.array([[1, 0], [0, 0]]), op2]
    with pytest.raises(ValueError, match='Inconsistent Kraus operator shapes'):
        _ = cirq.KrausChannel(kraus_ops=ops, key='m')