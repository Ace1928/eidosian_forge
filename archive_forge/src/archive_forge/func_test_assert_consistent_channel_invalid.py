import pytest
import numpy as np
import cirq
def test_assert_consistent_channel_invalid():
    channel = cirq.KrausChannel(kraus_ops=(np.array([[1, 1], [0, 0]]), np.array([[1, 0], [0, 0]])))
    with pytest.raises(AssertionError, match='cirq.KrausChannel.*2 1'):
        cirq.testing.assert_consistent_channel(channel)