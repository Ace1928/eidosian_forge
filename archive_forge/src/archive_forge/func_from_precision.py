from functools import cached_property
import numpy as np
from scipy import linalg
from scipy.stats import _multivariate
@staticmethod
def from_precision(precision, covariance=None):
    """
        Return a representation of a covariance from its precision matrix.

        Parameters
        ----------
        precision : array_like
            The precision matrix; that is, the inverse of a square, symmetric,
            positive definite covariance matrix.
        covariance : array_like, optional
            The square, symmetric, positive definite covariance matrix. If not
            provided, this may need to be calculated (e.g. to evaluate the
            cumulative distribution function of
            `scipy.stats.multivariate_normal`) by inverting `precision`.

        Notes
        -----
        Let the covariance matrix be :math:`A`, its precision matrix be
        :math:`P = A^{-1}`, and :math:`L` be the lower Cholesky factor such
        that :math:`L L^T = P`.
        Whitening of a data point :math:`x` is performed by computing
        :math:`x^T L`. :math:`\\log\\det{A}` is calculated as
        :math:`-2tr(\\log{L})`, where the :math:`\\log` operation is performed
        element-wise.

        This `Covariance` class does not support singular covariance matrices
        because the precision matrix does not exist for a singular covariance
        matrix.

        Examples
        --------
        Prepare a symmetric positive definite precision matrix ``P`` and a
        data point ``x``. (If the precision matrix is not already available,
        consider the other factory methods of the `Covariance` class.)

        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> n = 5
        >>> P = rng.random(size=(n, n))
        >>> P = P @ P.T  # a precision matrix must be positive definite
        >>> x = rng.random(size=n)

        Create the `Covariance` object.

        >>> cov = stats.Covariance.from_precision(P)

        Compare the functionality of the `Covariance` object against
        reference implementations.

        >>> res = cov.whiten(x)
        >>> ref = x @ np.linalg.cholesky(P)
        >>> np.allclose(res, ref)
        True
        >>> res = cov.log_pdet
        >>> ref = -np.linalg.slogdet(P)[-1]
        >>> np.allclose(res, ref)
        True

        """
    return CovViaPrecision(precision, covariance)