from functools import partial
from itertools import chain
import numpy as np
import pytest
from sklearn.metrics.cluster import (
from sklearn.utils._testing import assert_allclose
@pytest.mark.parametrize('metric', SUPERVISED_METRICS.values())
def test_single_sample(metric):
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        metric([i], [j])