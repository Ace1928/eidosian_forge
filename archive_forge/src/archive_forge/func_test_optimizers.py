import os
from collections import Counter
from itertools import combinations, product
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
from scipy.spatial import distance
from scipy.stats import shapiro
from scipy.stats._sobol import _test_find_index
from scipy.stats import qmc
from scipy.stats._qmc import (
@pytest.mark.parametrize('optimization,metric', [('random-CD', qmc.discrepancy), ('lloyd', lambda sample: -_l1_norm(sample))])
def test_optimizers(self, optimization, metric):
    engine = self.engine(d=2, scramble=False)
    sample_ref = engine.random(n=64)
    metric_ref = metric(sample_ref)
    optimal_ = self.engine(d=2, scramble=False, optimization=optimization)
    sample_ = optimal_.random(n=64)
    metric_ = metric(sample_)
    assert metric_ < metric_ref