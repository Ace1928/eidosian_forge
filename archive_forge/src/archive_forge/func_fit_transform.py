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
@_fit_context(prefer_skip_nested_validation=True)
def fit_transform(self, y):
    """Fit the label sets binarizer and transform the given label sets.

        Parameters
        ----------
        y : iterable of iterables
            A set of labels (any orderable and hashable object) for each
            sample. If the `classes` parameter is set, `y` will not be
            iterated.

        Returns
        -------
        y_indicator : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            A matrix such that `y_indicator[i, j] = 1` iff `classes_[j]`
            is in `y[i]`, and 0 otherwise. Sparse matrix will be of CSR
            format.
        """
    if self.classes is not None:
        return self.fit(y).transform(y)
    self._cached_dict = None
    class_mapping = defaultdict(int)
    class_mapping.default_factory = class_mapping.__len__
    yt = self._transform(y, class_mapping)
    tmp = sorted(class_mapping, key=class_mapping.get)
    dtype = int if all((isinstance(c, int) for c in tmp)) else object
    class_mapping = np.empty(len(tmp), dtype=dtype)
    class_mapping[:] = tmp
    self.classes_, inverse = np.unique(class_mapping, return_inverse=True)
    yt.indices = np.array(inverse[yt.indices], dtype=yt.indices.dtype, copy=False)
    if not self.sparse_output:
        yt = yt.toarray()
    return yt