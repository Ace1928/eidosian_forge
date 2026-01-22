import warnings
from collections.abc import Sequence
from itertools import chain
import numpy as np
from scipy.sparse import issparse
from ..utils._array_api import get_namespace
from ..utils.fixes import VisibleDeprecationWarning
from .validation import _assert_all_finite, check_array
def check_classification_targets(y):
    """Ensure that target y is of a non-regression type.

    Only the following target types (as defined in type_of_target) are allowed:
        'binary', 'multiclass', 'multiclass-multioutput',
        'multilabel-indicator', 'multilabel-sequences'

    Parameters
    ----------
    y : array-like
        Target values.
    """
    y_type = type_of_target(y, input_name='y')
    if y_type not in ['binary', 'multiclass', 'multiclass-multioutput', 'multilabel-indicator', 'multilabel-sequences']:
        raise ValueError(f'Unknown label type: {y_type}. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.')