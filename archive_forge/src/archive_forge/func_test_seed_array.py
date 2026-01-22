import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
def test_seed_array(self):
    if self.seed_vector_bits is None:
        bitgen_name = self.bit_generator.__name__
        pytest.skip(f'Vector seeding is not supported by {bitgen_name}')
    if self.seed_vector_bits == 32:
        dtype = np.uint32
    else:
        dtype = np.uint64
    seed = np.array([1], dtype=dtype)
    bg = self.bit_generator(seed)
    state1 = bg.state
    bg = self.bit_generator(1)
    state2 = bg.state
    assert_(comp_state(state1, state2))
    seed = np.arange(4, dtype=dtype)
    bg = self.bit_generator(seed)
    state1 = bg.state
    bg = self.bit_generator(seed[0])
    state2 = bg.state
    assert_(not comp_state(state1, state2))
    seed = np.arange(1500, dtype=dtype)
    bg = self.bit_generator(seed)
    state1 = bg.state
    bg = self.bit_generator(seed[0])
    state2 = bg.state
    assert_(not comp_state(state1, state2))
    seed = 2 ** np.mod(np.arange(1500, dtype=dtype), self.seed_vector_bits - 1) + 1
    bg = self.bit_generator(seed)
    state1 = bg.state
    bg = self.bit_generator(seed[0])
    state2 = bg.state
    assert_(not comp_state(state1, state2))