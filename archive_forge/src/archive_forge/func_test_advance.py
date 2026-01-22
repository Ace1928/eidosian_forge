import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
def test_advance(self):
    state = self.rg.bit_generator.state
    if hasattr(self.rg.bit_generator, 'advance'):
        self.rg.bit_generator.advance(self.advance)
        assert_(not comp_state(state, self.rg.bit_generator.state))
    else:
        bitgen_name = self.rg.bit_generator.__class__.__name__
        pytest.skip(f'Advance is not supported by {bitgen_name}')