import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_p_non_contiguous(self):
    p = np.arange(15.0)
    p /= np.sum(p[1::3])
    pvals = p[1::3]
    random.seed(1432985819)
    non_contig = random.multinomial(100, pvals=pvals)
    random.seed(1432985819)
    contig = random.multinomial(100, pvals=np.ascontiguousarray(pvals))
    assert_array_equal(non_contig, contig)