import warnings
from collections.abc import Sequence
from itertools import chain
import numpy as np
from scipy.sparse import issparse
from ..utils._array_api import get_namespace
from ..utils.fixes import VisibleDeprecationWarning
from .validation import _assert_all_finite, check_array
def unique_labels(*ys):
    """Extract an ordered array of unique labels.

    We don't allow:
        - mix of multilabel and multiclass (single label) targets
        - mix of label indicator matrix and anything else,
          because there are no explicit labels)
        - mix of label indicator matrices of different sizes
        - mix of string and integer labels

    At the moment, we also don't allow "multiclass-multioutput" input type.

    Parameters
    ----------
    *ys : array-likes
        Label values.

    Returns
    -------
    out : ndarray of shape (n_unique_labels,)
        An ordered array of unique labels.

    Examples
    --------
    >>> from sklearn.utils.multiclass import unique_labels
    >>> unique_labels([3, 5, 5, 5, 7, 7])
    array([3, 5, 7])
    >>> unique_labels([1, 2, 3, 4], [2, 2, 3, 4])
    array([1, 2, 3, 4])
    >>> unique_labels([1, 2, 10], [5, 11])
    array([ 1,  2,  5, 10, 11])
    """
    xp, is_array_api_compliant = get_namespace(*ys)
    if not ys:
        raise ValueError('No argument has been passed.')
    ys_types = set((type_of_target(x) for x in ys))
    if ys_types == {'binary', 'multiclass'}:
        ys_types = {'multiclass'}
    if len(ys_types) > 1:
        raise ValueError('Mix type of y not allowed, got types %s' % ys_types)
    label_type = ys_types.pop()
    if label_type == 'multilabel-indicator' and len(set((check_array(y, accept_sparse=['csr', 'csc', 'coo']).shape[1] for y in ys))) > 1:
        raise ValueError('Multi-label binary indicator input with different numbers of labels')
    _unique_labels = _FN_UNIQUE_LABELS.get(label_type, None)
    if not _unique_labels:
        raise ValueError('Unknown label type: %s' % repr(ys))
    if is_array_api_compliant:
        unique_ys = xp.concat([_unique_labels(y) for y in ys])
        return xp.unique_values(unique_ys)
    ys_labels = set(chain.from_iterable(((i for i in _unique_labels(y)) for y in ys)))
    if len(set((isinstance(label, str) for label in ys_labels))) > 1:
        raise ValueError('Mix of label input types (string and number)')
    return xp.asarray(sorted(ys_labels))