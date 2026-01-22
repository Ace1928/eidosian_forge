import itertools
import numpy as np
import pytest
from sklearn import datasets
from sklearn.covariance import MinCovDet, empirical_covariance, fast_mcd
from sklearn.utils._testing import assert_array_almost_equal
def test_mcd(global_random_seed):
    launch_mcd_on_dataset(100, 5, 0, 0.02, 0.1, 75, global_random_seed)
    launch_mcd_on_dataset(100, 5, 20, 0.3, 0.3, 65, global_random_seed)
    launch_mcd_on_dataset(100, 5, 40, 0.1, 0.1, 50, global_random_seed)
    launch_mcd_on_dataset(1000, 5, 450, 0.1, 0.1, 540, global_random_seed)
    launch_mcd_on_dataset(1700, 5, 800, 0.1, 0.1, 870, global_random_seed)
    launch_mcd_on_dataset(500, 1, 100, 0.02, 0.02, 350, global_random_seed)