import numpy as np
from numpy.testing import assert_array_equal, assert_array_compare
from numpy.random import SeedSequence
def test_zero_padding():
    """ Ensure that the implicit zero-padding does not cause problems.
    """
    ss0 = SeedSequence(42)
    ss1 = SeedSequence(42 << 32)
    assert_array_compare(np.not_equal, ss0.generate_state(4), ss1.generate_state(4))
    expected42 = np.array([3444837047, 2669555309, 2046530742, 3581440988], dtype=np.uint32)
    assert_array_equal(SeedSequence(42).generate_state(4), expected42)
    assert_array_compare(np.not_equal, SeedSequence(42, spawn_key=(0,)).generate_state(4), expected42)