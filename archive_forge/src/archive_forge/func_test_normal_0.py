import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_normal_0(self):
    assert_equal(random.normal(scale=0), 0)
    assert_raises(ValueError, random.normal, scale=-0.0)