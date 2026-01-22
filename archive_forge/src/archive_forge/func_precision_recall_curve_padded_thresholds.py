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
def precision_recall_curve_padded_thresholds(*args, **kwargs):
    """
    The dimensions of precision-recall pairs and the threshold array as
    returned by the precision_recall_curve do not match. See
    func:`sklearn.metrics.precision_recall_curve`

    This prevents implicit conversion of return value triple to an higher
    dimensional np.array of dtype('float64') (it will be of dtype('object)
    instead). This again is needed for assert_array_equal to work correctly.

    As a workaround we pad the threshold array with NaN values to match
    the dimension of precision and recall arrays respectively.
    """
    precision, recall, thresholds = precision_recall_curve(*args, **kwargs)
    pad_threshholds = len(precision) - len(thresholds)
    return np.array([precision, recall, np.pad(thresholds.astype(np.float64), pad_width=(0, pad_threshholds), mode='constant', constant_values=[np.nan])])