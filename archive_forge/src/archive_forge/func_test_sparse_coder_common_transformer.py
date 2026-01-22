import itertools
import warnings
from functools import partial
import numpy as np
import pytest
import sklearn
from sklearn.base import clone
from sklearn.decomposition import (
from sklearn.decomposition._dict_learning import _update_dict
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.parallel import Parallel
def test_sparse_coder_common_transformer():
    rng = np.random.RandomState(777)
    n_components, n_features = (40, 3)
    init_dict = rng.rand(n_components, n_features)
    sc = SparseCoder(init_dict)
    check_transformer_data_not_an_array(sc.__class__.__name__, sc)
    check_transformer_general(sc.__class__.__name__, sc)
    check_transformer_general_memmap = partial(check_transformer_general, readonly_memmap=True)
    check_transformer_general_memmap(sc.__class__.__name__, sc)
    check_transformers_unfitted(sc.__class__.__name__, sc)