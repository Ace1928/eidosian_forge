import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
@pytest.fixture(scope='module', params=INT_FUNCS)
def int_func(request):
    return (request.param, INT_FUNCS[request.param], INT_FUNC_HASHES[request.param])