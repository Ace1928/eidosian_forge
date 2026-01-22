import numpy as np
import pytest
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_one_feature():
    X = np.random.normal(size=(128, 1))
    BisectingKMeans(bisecting_strategy='biggest_inertia', random_state=0).fit(X)