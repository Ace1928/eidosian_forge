import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.murmurhash import murmurhash3_32
def test_mmhash3_int_array():
    rng = np.random.RandomState(42)
    keys = rng.randint(-5342534, 345345, size=3 * 2 * 1).astype(np.int32)
    keys = keys.reshape((3, 2, 1))
    for seed in [0, 42]:
        expected = np.array([murmurhash3_32(int(k), seed) for k in keys.flat])
        expected = expected.reshape(keys.shape)
        assert_array_equal(murmurhash3_32(keys, seed), expected)
    for seed in [0, 42]:
        expected = np.array([murmurhash3_32(k, seed, positive=True) for k in keys.flat])
        expected = expected.reshape(keys.shape)
        assert_array_equal(murmurhash3_32(keys, seed, positive=True), expected)