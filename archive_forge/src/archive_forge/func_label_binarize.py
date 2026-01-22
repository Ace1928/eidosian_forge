import array
import itertools
import warnings
from collections import defaultdict
from numbers import Integral
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import column_or_1d
from ..utils._encode import _encode, _unique
from ..utils._param_validation import Interval, validate_params
from ..utils.multiclass import type_of_target, unique_labels
from ..utils.sparsefuncs import min_max_axis
from ..utils.validation import _num_samples, check_array, check_is_fitted
@validate_params({'y': ['array-like'], 'classes': ['array-like'], 'neg_label': [Interval(Integral, None, None, closed='neither')], 'pos_label': [Interval(Integral, None, None, closed='neither')], 'sparse_output': ['boolean']}, prefer_skip_nested_validation=True)
def label_binarize(y, *, classes, neg_label=0, pos_label=1, sparse_output=False):
    """Binarize labels in a one-vs-all fashion.

    Several regression and binary classification algorithms are
    available in scikit-learn. A simple way to extend these algorithms
    to the multi-class classification case is to use the so-called
    one-vs-all scheme.

    This function makes it possible to compute this transformation for a
    fixed set of class labels known ahead of time.

    Parameters
    ----------
    y : array-like
        Sequence of integer labels or multilabel data to encode.

    classes : array-like of shape (n_classes,)
        Uniquely holds the label for each class.

    neg_label : int, default=0
        Value with which negative labels must be encoded.

    pos_label : int, default=1
        Value with which positive labels must be encoded.

    sparse_output : bool, default=False,
        Set to true if output binary array is desired in CSR sparse format.

    Returns
    -------
    Y : {ndarray, sparse matrix} of shape (n_samples, n_classes)
        Shape will be (n_samples, 1) for binary problems. Sparse matrix will
        be of CSR format.

    See Also
    --------
    LabelBinarizer : Class used to wrap the functionality of label_binarize and
        allow for fitting to classes independently of the transform operation.

    Examples
    --------
    >>> from sklearn.preprocessing import label_binarize
    >>> label_binarize([1, 6], classes=[1, 2, 4, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

    The class ordering is preserved:

    >>> label_binarize([1, 6], classes=[1, 6, 4, 2])
    array([[1, 0, 0, 0],
           [0, 1, 0, 0]])

    Binary targets transform to a column vector

    >>> label_binarize(['yes', 'no', 'no', 'yes'], classes=['no', 'yes'])
    array([[1],
           [0],
           [0],
           [1]])
    """
    if not isinstance(y, list):
        y = check_array(y, input_name='y', accept_sparse='csr', ensure_2d=False, dtype=None)
    elif _num_samples(y) == 0:
        raise ValueError('y has 0 samples: %r' % y)
    if neg_label >= pos_label:
        raise ValueError('neg_label={0} must be strictly less than pos_label={1}.'.format(neg_label, pos_label))
    if sparse_output and (pos_label == 0 or neg_label != 0):
        raise ValueError('Sparse binarization is only supported with non zero pos_label and zero neg_label, got pos_label={0} and neg_label={1}'.format(pos_label, neg_label))
    pos_switch = pos_label == 0
    if pos_switch:
        pos_label = -neg_label
    y_type = type_of_target(y)
    if 'multioutput' in y_type:
        raise ValueError('Multioutput target data is not supported with label binarization')
    if y_type == 'unknown':
        raise ValueError('The type of target data is not known')
    n_samples = y.shape[0] if sp.issparse(y) else len(y)
    n_classes = len(classes)
    classes = np.asarray(classes)
    if y_type == 'binary':
        if n_classes == 1:
            if sparse_output:
                return sp.csr_matrix((n_samples, 1), dtype=int)
            else:
                Y = np.zeros((len(y), 1), dtype=int)
                Y += neg_label
                return Y
        elif len(classes) >= 3:
            y_type = 'multiclass'
    sorted_class = np.sort(classes)
    if y_type == 'multilabel-indicator':
        y_n_classes = y.shape[1] if hasattr(y, 'shape') else len(y[0])
        if classes.size != y_n_classes:
            raise ValueError('classes {0} mismatch with the labels {1} found in the data'.format(classes, unique_labels(y)))
    if y_type in ('binary', 'multiclass'):
        y = column_or_1d(y)
        y_in_classes = np.isin(y, classes)
        y_seen = y[y_in_classes]
        indices = np.searchsorted(sorted_class, y_seen)
        indptr = np.hstack((0, np.cumsum(y_in_classes)))
        data = np.empty_like(indices)
        data.fill(pos_label)
        Y = sp.csr_matrix((data, indices, indptr), shape=(n_samples, n_classes))
    elif y_type == 'multilabel-indicator':
        Y = sp.csr_matrix(y)
        if pos_label != 1:
            data = np.empty_like(Y.data)
            data.fill(pos_label)
            Y.data = data
    else:
        raise ValueError('%s target data is not supported with label binarization' % y_type)
    if not sparse_output:
        Y = Y.toarray()
        Y = Y.astype(int, copy=False)
        if neg_label != 0:
            Y[Y == 0] = neg_label
        if pos_switch:
            Y[Y == pos_label] = 0
    else:
        Y.data = Y.data.astype(int, copy=False)
    if np.any(classes != sorted_class):
        indices = np.searchsorted(sorted_class, classes)
        Y = Y[:, indices]
    if y_type == 'binary':
        if sparse_output:
            Y = Y.getcol(-1)
        else:
            Y = Y[:, -1].reshape((-1, 1))
    return Y