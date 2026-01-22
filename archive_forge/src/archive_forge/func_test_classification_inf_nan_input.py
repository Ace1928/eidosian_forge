from functools import partial
from inspect import signature
from itertools import chain, permutations, product
import numpy as np
import pytest
from sklearn._config import config_context
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import (
from sklearn.metrics._base import _average_binary_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples, check_random_state
@pytest.mark.parametrize('metric', CLASSIFICATION_METRICS.values())
@pytest.mark.parametrize('y_true, y_score', invalids_nan_inf + [([np.nan, 1, 2], [1, 2, 3]), ([np.inf, 1, 2], [1, 2, 3])])
def test_classification_inf_nan_input(metric, y_true, y_score):
    """check that classification metrics raise a message mentioning the
    occurrence of non-finite values in the target vectors."""
    if not np.isfinite(y_true).all():
        input_name = 'y_true'
        if np.isnan(y_true).any():
            unexpected_value = 'NaN'
        else:
            unexpected_value = 'infinity or a value too large'
    else:
        input_name = 'y_pred'
        if np.isnan(y_score).any():
            unexpected_value = 'NaN'
        else:
            unexpected_value = 'infinity or a value too large'
    err_msg = f'Input {input_name} contains {unexpected_value}'
    with pytest.raises(ValueError, match=err_msg):
        metric(y_true, y_score)