import warnings
import numpy as np
import pytest
from sklearn.cluster import MeanShift, estimate_bandwidth, get_bin_seeds, mean_shift
from sklearn.datasets import make_blobs
from sklearn.metrics import v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
def test_meanshift_all_orphans():
    ms = MeanShift(bandwidth=0.1, seeds=[[-9, -9], [-10, -10]])
    msg = 'No point was within bandwidth=0.1'
    with pytest.raises(ValueError, match=msg):
        ms.fit(X)