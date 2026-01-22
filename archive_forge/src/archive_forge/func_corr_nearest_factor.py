import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from scipy.optimize import fminbound
import warnings
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (
def corr_nearest_factor(corr, rank, ctol=1e-06, lam_min=1e-30, lam_max=1e+30, maxiter=1000):
    """
    Find the nearest correlation matrix with factor structure to a
    given square matrix.

    Parameters
    ----------
    corr : square array
        The target matrix (to which the nearest correlation matrix is
        sought).  Must be square, but need not be positive
        semidefinite.
    rank : int
        The rank of the factor structure of the solution, i.e., the
        number of linearly independent columns of X.
    ctol : positive real
        Convergence criterion.
    lam_min : float
        Tuning parameter for spectral projected gradient optimization
        (smallest allowed step in the search direction).
    lam_max : float
        Tuning parameter for spectral projected gradient optimization
        (largest allowed step in the search direction).
    maxiter : int
        Maximum number of iterations in spectral projected gradient
        optimization.

    Returns
    -------
    rslt : Bunch
        rslt.corr is a FactoredPSDMatrix defining the estimated
        correlation structure.  Other fields of `rslt` contain
        returned values from spg_optim.

    Notes
    -----
    A correlation matrix has factor structure if it can be written in
    the form I + XX' - diag(XX'), where X is n x k with linearly
    independent columns, and with each row having sum of squares at
    most equal to 1.  The approximation is made in terms of the
    Frobenius norm.

    This routine is useful when one has an approximate correlation
    matrix that is not positive semidefinite, and there is need to
    estimate the inverse, square root, or inverse square root of the
    population correlation matrix.  The factor structure allows these
    tasks to be done without constructing any n x n matrices.

    This is a non-convex problem with no known guaranteed globally
    convergent algorithm for computing the solution.  Borsdof, Higham
    and Raydan (2010) compared several methods for this problem and
    found the spectral projected gradient (SPG) method (used here) to
    perform best.

    The input matrix `corr` can be a dense numpy array or any scipy
    sparse matrix.  The latter is useful if the input matrix is
    obtained by thresholding a very large sample correlation matrix.
    If `corr` is sparse, the calculations are optimized to save
    memory, so no working matrix with more than 10^6 elements is
    constructed.

    References
    ----------
    .. [*] R Borsdof, N Higham, M Raydan (2010).  Computing a nearest
       correlation matrix with factor structure. SIAM J Matrix Anal Appl,
       31:5, 2603-2622.
       http://eprints.ma.man.ac.uk/1523/01/covered/MIMS_ep2009_87.pdf

    Examples
    --------
    Hard thresholding a correlation matrix may result in a matrix that
    is not positive semidefinite.  We can approximate a hard
    thresholded correlation matrix with a PSD matrix as follows, where
    `corr` is the input correlation matrix.

    >>> import numpy as np
    >>> from statsmodels.stats.correlation_tools import corr_nearest_factor
    >>> np.random.seed(1234)
    >>> b = 1.5 - np.random.rand(10, 1)
    >>> x = np.random.randn(100,1).dot(b.T) + np.random.randn(100,10)
    >>> corr = np.corrcoef(x.T)
    >>> corr = corr * (np.abs(corr) >= 0.3)
    >>> rslt = corr_nearest_factor(corr, 3)
    """
    p, _ = corr.shape
    u, s, vt = svds(corr, rank)
    X = u * np.sqrt(s)
    nm = np.sqrt((X ** 2).sum(1))
    ii = np.flatnonzero(nm > 1e-05)
    X[ii, :] /= nm[ii][:, None]
    corr1 = corr.copy()
    if type(corr1) is np.ndarray:
        np.fill_diagonal(corr1, 0)
    elif sparse.issparse(corr1):
        corr1.setdiag(np.zeros(corr1.shape[0]))
        corr1.eliminate_zeros()
        corr1.sort_indices()
    else:
        raise ValueError('Matrix type not supported')

    def grad(X):
        gr = np.dot(X, np.dot(X.T, X))
        if type(corr1) is np.ndarray:
            gr -= np.dot(corr1, X)
        else:
            gr -= corr1.dot(X)
        gr -= (X * X).sum(1)[:, None] * X
        return 4 * gr

    def func(X):
        if type(corr1) is np.ndarray:
            M = np.dot(X, X.T)
            np.fill_diagonal(M, 0)
            M -= corr1
            fval = (M * M).sum()
            return fval
        else:
            fval = 0.0
            max_ws = 1000000.0
            bs = int(max_ws / X.shape[0])
            ir = 0
            while ir < X.shape[0]:
                ir2 = min(ir + bs, X.shape[0])
                u = np.dot(X[ir:ir2, :], X.T)
                ii = np.arange(u.shape[0])
                u[ii, ir + ii] = 0
                u -= np.asarray(corr1[ir:ir2, :].todense())
                fval += (u * u).sum()
                ir += bs
            return fval
    rslt = _spg_optim(func, grad, X, _project_correlation_factors, ctol=ctol, lam_min=lam_min, lam_max=lam_max, maxiter=maxiter)
    root = rslt.params
    diag = 1 - (root ** 2).sum(1)
    soln = FactoredPSDMatrix(diag, root)
    rslt.corr = soln
    del rslt.params
    return rslt