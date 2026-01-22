import numpy as np
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
@pytest.mark.parametrize('reps', range(1, 100, 7))
def test_pack_bits(reps):
    data = np.random.randint(2, size=reps, dtype=bool)
    packed = v2.pack_bits(data)
    assert isinstance(packed, bytes)
    assert len(packed) == (reps + 7) // 8
    unpacked = v2.unpack_bits(packed, reps)
    np.testing.assert_array_equal(unpacked, data)