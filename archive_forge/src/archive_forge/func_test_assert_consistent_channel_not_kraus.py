import pytest
import numpy as np
import cirq
def test_assert_consistent_channel_not_kraus():
    with pytest.raises(AssertionError, match='12.*has_kraus'):
        cirq.testing.assert_consistent_channel(12)