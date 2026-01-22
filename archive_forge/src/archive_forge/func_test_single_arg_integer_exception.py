import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
@pytest.mark.parametrize('high', [-2, [-2]])
@pytest.mark.parametrize('endpoint', [True, False])
def test_single_arg_integer_exception(high, endpoint):
    gen = Generator(MT19937(0))
    msg = 'high < 0' if endpoint else 'high <= 0'
    with pytest.raises(ValueError, match=msg):
        gen.integers(high, endpoint=endpoint)
    msg = 'low > high' if endpoint else 'low >= high'
    with pytest.raises(ValueError, match=msg):
        gen.integers(-1, high, endpoint=endpoint)
    with pytest.raises(ValueError, match=msg):
        gen.integers([-1], high, endpoint=endpoint)