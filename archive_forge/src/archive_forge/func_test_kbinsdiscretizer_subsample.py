import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
@pytest.mark.parametrize('strategy', ['uniform', 'kmeans', 'quantile'])
def test_kbinsdiscretizer_subsample(strategy, global_random_seed):
    X = np.random.RandomState(global_random_seed).random_sample((100000, 1)) + 1
    kbd_subsampling = KBinsDiscretizer(strategy=strategy, subsample=50000, random_state=global_random_seed)
    kbd_subsampling.fit(X)
    kbd_no_subsampling = clone(kbd_subsampling)
    kbd_no_subsampling.set_params(subsample=None)
    kbd_no_subsampling.fit(X)
    assert_allclose(kbd_subsampling.bin_edges_[0], kbd_no_subsampling.bin_edges_[0], rtol=0.01)