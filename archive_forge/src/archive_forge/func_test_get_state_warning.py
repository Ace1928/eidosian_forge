import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_get_state_warning(self):
    rs = random.RandomState(PCG64())
    with suppress_warnings() as sup:
        w = sup.record(RuntimeWarning)
        state = rs.get_state()
        assert_(len(w) == 1)
        assert isinstance(state, dict)
        assert state['bit_generator'] == 'PCG64'