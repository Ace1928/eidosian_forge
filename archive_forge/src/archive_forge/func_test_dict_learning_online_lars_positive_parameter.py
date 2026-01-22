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
def test_dict_learning_online_lars_positive_parameter():
    err_msg = "Positive constraint not supported for 'lars' coding method."
    with pytest.raises(ValueError, match=err_msg):
        dict_learning_online(X, batch_size=4, max_iter=10, positive_code=True)