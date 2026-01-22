import itertools
import sys
import time
from numbers import Integral, Real
from warnings import warn
import numpy as np
from joblib import effective_n_jobs
from scipy import linalg
from ..base import (
from ..linear_model import Lars, Lasso, LassoLars, orthogonal_mp_gram
from ..utils import check_array, check_random_state, gen_batches, gen_even_slices
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.extmath import randomized_svd, row_norms, svd_flip
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted
@validate_params({'X': ['array-like'], 'method': [StrOptions({'lars', 'cd'})], 'return_n_iter': ['boolean'], 'method_max_iter': [Interval(Integral, 0, None, closed='left')]}, prefer_skip_nested_validation=False)
def dict_learning(X, n_components, *, alpha, max_iter=100, tol=1e-08, method='lars', n_jobs=None, dict_init=None, code_init=None, callback=None, verbose=False, random_state=None, return_n_iter=False, positive_dict=False, positive_code=False, method_max_iter=1000):
    """Solve a dictionary learning matrix factorization problem.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_Fro^2 + alpha * || U ||_1,1
                     (U,V)
                    with || V_k ||_2 = 1 for all  0 <= k < n_components

    where V is the dictionary and U is the sparse code. ||.||_Fro stands for
    the Frobenius norm and ||.||_1,1 stands for the entry-wise matrix norm
    which is the sum of the absolute values of all the entries in the matrix.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.

    n_components : int
        Number of dictionary atoms to extract.

    alpha : int or float
        Sparsity controlling parameter.

    max_iter : int, default=100
        Maximum number of iterations to perform.

    tol : float, default=1e-8
        Tolerance for the stopping condition.

    method : {'lars', 'cd'}, default='lars'
        The method used:

        * `'lars'`: uses the least angle regression method to solve the lasso
           problem (`linear_model.lars_path`);
        * `'cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). Lars will be faster if
          the estimated components are sparse.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    dict_init : ndarray of shape (n_components, n_features), default=None
        Initial value for the dictionary for warm restart scenarios. Only used
        if `code_init` and `dict_init` are not None.

    code_init : ndarray of shape (n_samples, n_components), default=None
        Initial value for the sparse code for warm restart scenarios. Only used
        if `code_init` and `dict_init` are not None.

    callback : callable, default=None
        Callable that gets invoked every five iterations.

    verbose : bool, default=False
        To control the verbosity of the procedure.

    random_state : int, RandomState instance or None, default=None
        Used for randomly initializing the dictionary. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    positive_dict : bool, default=False
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    positive_code : bool, default=False
        Whether to enforce positivity when finding the code.

        .. versionadded:: 0.20

    method_max_iter : int, default=1000
        Maximum number of iterations to perform.

        .. versionadded:: 0.22

    Returns
    -------
    code : ndarray of shape (n_samples, n_components)
        The sparse code factor in the matrix factorization.

    dictionary : ndarray of shape (n_components, n_features),
        The dictionary factor in the matrix factorization.

    errors : array
        Vector of errors at each iteration.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to True.

    See Also
    --------
    dict_learning_online : Solve a dictionary learning matrix factorization
        problem online.
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    MiniBatchDictionaryLearning : A faster, less accurate version
        of the dictionary learning algorithm.
    SparsePCA : Sparse Principal Components Analysis.
    MiniBatchSparsePCA : Mini-batch Sparse Principal Components Analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_sparse_coded_signal
    >>> from sklearn.decomposition import dict_learning
    >>> X, _, _ = make_sparse_coded_signal(
    ...     n_samples=30, n_components=15, n_features=20, n_nonzero_coefs=10,
    ...     random_state=42,
    ... )
    >>> U, V, errors = dict_learning(X, n_components=15, alpha=0.1, random_state=42)

    We can check the level of sparsity of `U`:

    >>> np.mean(U == 0)
    0.6...

    We can compare the average squared euclidean norm of the reconstruction
    error of the sparse coded signal relative to the squared euclidean norm of
    the original signal:

    >>> X_hat = U @ V
    >>> np.mean(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X ** 2, axis=1))
    0.01...
    """
    estimator = DictionaryLearning(n_components=n_components, alpha=alpha, max_iter=max_iter, tol=tol, fit_algorithm=method, n_jobs=n_jobs, dict_init=dict_init, callback=callback, code_init=code_init, verbose=verbose, random_state=random_state, positive_code=positive_code, positive_dict=positive_dict, transform_max_iter=method_max_iter).set_output(transform='default')
    code = estimator.fit_transform(X)
    if return_n_iter:
        return (code, estimator.components_, estimator.error_, estimator.n_iter_)
    return (code, estimator.components_, estimator.error_)