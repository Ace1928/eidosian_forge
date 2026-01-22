from functools import partial
from itertools import chain
import numpy as np
import pytest
from sklearn.metrics.cluster import (
from sklearn.utils._testing import assert_allclose
def test_symmetric_non_symmetric_union():
    assert sorted(SYMMETRIC_METRICS + NON_SYMMETRIC_METRICS) == sorted(SUPERVISED_METRICS)