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
@validate_params({'n_dim': [Interval(Integral, 1, None, closed='left')], 'random_state': ['random_state']}, prefer_skip_nested_validation=True)
def make_spd_matrix(n_dim, *, random_state=None):
    """Generate a random symmetric, positive-definite matrix.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_dim : int
        The matrix dimension.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_dim, n_dim)
        The random symmetric, positive-definite matrix.

    See Also
    --------
    make_sparse_spd_matrix: Generate a sparse symmetric definite positive matrix.

    Examples
    --------
    >>> from sklearn.datasets import make_spd_matrix
    >>> make_spd_matrix(n_dim=2, random_state=42)
    array([[2.09..., 0.34...],
           [0.34..., 0.21...]])
    """
    generator = check_random_state(random_state)
    A = generator.uniform(size=(n_dim, n_dim))
    U, _, Vt = linalg.svd(np.dot(A.T, A), check_finite=False)
    X = np.dot(np.dot(U, 1.0 + np.diag(generator.uniform(size=n_dim))), Vt)
    return X