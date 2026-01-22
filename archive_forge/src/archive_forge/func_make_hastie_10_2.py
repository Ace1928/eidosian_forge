import array
import numbers
import warnings
from collections.abc import Iterable
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from ..preprocessing import MultiLabelBinarizer
from ..utils import check_array, check_random_state
from ..utils import shuffle as util_shuffle
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.random import sample_without_replacement
@validate_params({'n_samples': [Interval(Integral, 1, None, closed='left')], 'random_state': ['random_state']}, prefer_skip_nested_validation=True)
def make_hastie_10_2(n_samples=12000, *, random_state=None):
    """Generate data for binary classification used in Hastie et al. 2009, Example 10.2.

    The ten features are standard independent Gaussian and
    the target ``y`` is defined by::

      y[i] = 1 if np.sum(X[i] ** 2) > 9.34 else -1

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=12000
        The number of samples.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, 10)
        The input samples.

    y : ndarray of shape (n_samples,)
        The output values.

    See Also
    --------
    make_gaussian_quantiles : A generalization of this dataset approach.

    References
    ----------
    .. [1] T. Hastie, R. Tibshirani and J. Friedman, "Elements of Statistical
           Learning Ed. 2", Springer, 2009.
    """
    rs = check_random_state(random_state)
    shape = (n_samples, 10)
    X = rs.normal(size=shape).reshape(shape)
    y = ((X ** 2.0).sum(axis=1) > 9.34).astype(np.float64, copy=False)
    y[y == 0.0] = -1.0
    return (X, y)