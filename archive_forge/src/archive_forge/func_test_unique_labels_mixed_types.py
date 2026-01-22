from itertools import product
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import config_context, datasets
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.utils._array_api import yield_namespace_device_dtype_combinations
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.multiclass import (
def test_unique_labels_mixed_types():
    mix_clf_format = product(EXAMPLES['multilabel-indicator'], EXAMPLES['multiclass'] + EXAMPLES['binary'])
    for y_multilabel, y_multiclass in mix_clf_format:
        with pytest.raises(ValueError):
            unique_labels(y_multiclass, y_multilabel)
        with pytest.raises(ValueError):
            unique_labels(y_multilabel, y_multiclass)
    with pytest.raises(ValueError):
        unique_labels([[1, 2]], [['a', 'd']])
    with pytest.raises(ValueError):
        unique_labels(['1', 2])
    with pytest.raises(ValueError):
        unique_labels([['1', 2], [1, 3]])
    with pytest.raises(ValueError):
        unique_labels([['1', '2'], [2, 3]])