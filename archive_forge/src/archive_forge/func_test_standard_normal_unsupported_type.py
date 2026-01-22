import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_standard_normal_unsupported_type(self):
    assert_raises(TypeError, random.standard_normal, dtype=np.int32)