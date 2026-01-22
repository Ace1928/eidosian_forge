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
@validate_params({'n_dim': [Hidden(None), Interval(Integral, 1, None, closed='left')], 'alpha': [Interval(Real, 0, 1, closed='both')], 'norm_diag': ['boolean'], 'smallest_coef': [Interval(Real, 0, 1, closed='both')], 'largest_coef': [Interval(Real, 0, 1, closed='both')], 'sparse_format': [StrOptions({'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}), None], 'random_state': ['random_state'], 'dim': [Interval(Integral, 1, None, closed='left'), Hidden(StrOptions({'deprecated'}))]}, prefer_skip_nested_validation=True)
def make_sparse_spd_matrix(n_dim=None, *, alpha=0.95, norm_diag=False, smallest_coef=0.1, largest_coef=0.9, sparse_format=None, random_state=None, dim='deprecated'):
    """Generate a sparse symmetric definite positive matrix.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_dim : int, default=1
        The size of the random matrix to generate.

        .. versionchanged:: 1.4
            Renamed from ``dim`` to ``n_dim``.

    alpha : float, default=0.95
        The probability that a coefficient is zero (see notes). Larger values
        enforce more sparsity. The value should be in the range 0 and 1.

    norm_diag : bool, default=False
        Whether to normalize the output matrix to make the leading diagonal
        elements all 1.

    smallest_coef : float, default=0.1
        The value of the smallest coefficient between 0 and 1.

    largest_coef : float, default=0.9
        The value of the largest coefficient between 0 and 1.

    sparse_format : str, default=None
        String representing the output sparse format, such as 'csc', 'csr', etc.
        If ``None``, return a dense numpy ndarray.

        .. versionadded:: 1.4

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    dim : int, default=1
        The size of the random matrix to generate.

        .. deprecated:: 1.4
            `dim` is deprecated and will be removed in 1.6.

    Returns
    -------
    prec : ndarray or sparse matrix of shape (dim, dim)
        The generated matrix. If ``sparse_format=None``, this would be an ndarray.
        Otherwise, this will be a sparse matrix of the specified format.

    See Also
    --------
    make_spd_matrix : Generate a random symmetric, positive-definite matrix.

    Notes
    -----
    The sparsity is actually imposed on the cholesky factor of the matrix.
    Thus alpha does not translate directly into the filling fraction of
    the matrix itself.

    Examples
    --------
    >>> from sklearn.datasets import make_sparse_spd_matrix
    >>> make_sparse_spd_matrix(n_dim=4, norm_diag=False, random_state=42)
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])
    """
    random_state = check_random_state(random_state)
    if n_dim is not None and dim != 'deprecated':
        raise ValueError('`dim` and `n_dim` cannot be both specified. Please use `n_dim` only as `dim` is deprecated in v1.4 and will be removed in v1.6.')
    if dim != 'deprecated':
        warnings.warn('dim was deprecated in version 1.4 and will be removed in 1.6.Please use ``n_dim`` instead.', FutureWarning)
        _n_dim = dim
    elif n_dim is None:
        _n_dim = 1
    else:
        _n_dim = n_dim
    chol = -sp.eye(_n_dim)
    aux = sp.random(m=_n_dim, n=_n_dim, density=1 - alpha, data_rvs=lambda x: random_state.uniform(low=smallest_coef, high=largest_coef, size=x), random_state=random_state)
    aux = sp.tril(aux, k=-1, format='csc')
    permutation = random_state.permutation(_n_dim)
    aux = aux[permutation].T[permutation]
    chol += aux
    prec = chol.T @ chol
    if norm_diag:
        d = sp.diags(1.0 / np.sqrt(prec.diagonal()))
        prec = d @ prec @ d
    if sparse_format is None:
        return prec.toarray()
    else:
        return prec.asformat(sparse_format)