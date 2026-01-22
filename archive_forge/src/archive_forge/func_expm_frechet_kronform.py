import numpy as np
import scipy.linalg
def expm_frechet_kronform(A, method=None, check_finite=True):
    """
    Construct the Kronecker form of the Frechet derivative of expm.

    Parameters
    ----------
    A : array_like with shape (N, N)
        Matrix to be expm'd.
    method : str, optional
        Extra keyword to be passed to expm_frechet.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    K : 2-D ndarray with shape (N*N, N*N)
        Kronecker form of the Frechet derivative of the matrix exponential.

    Notes
    -----
    This function is used to help compute the condition number
    of the matrix exponential.

    See Also
    --------
    expm : Compute a matrix exponential.
    expm_frechet : Compute the Frechet derivative of the matrix exponential.
    expm_cond : Compute the relative condition number of the matrix exponential
                in the Frobenius norm.

    """
    if check_finite:
        A = np.asarray_chkfinite(A)
    else:
        A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')
    n = A.shape[0]
    ident = np.identity(n)
    cols = []
    for i in range(n):
        for j in range(n):
            E = np.outer(ident[i], ident[j])
            F = expm_frechet(A, E, method=method, compute_expm=False, check_finite=False)
            cols.append(vec(F))
    return np.vstack(cols).T