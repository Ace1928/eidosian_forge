import numpy
from numpy import asarray_chkfinite, single, asarray, array
from numpy.linalg import norm
from ._misc import LinAlgError, _datacopied
from .lapack import get_lapack_funcs
from ._decomp import eigvals
def rsf2csf(T, Z, check_finite=True):
    """
    Convert real Schur form to complex Schur form.

    Convert a quasi-diagonal real-valued Schur form to the upper-triangular
    complex-valued Schur form.

    Parameters
    ----------
    T : (M, M) array_like
        Real Schur form of the original array
    Z : (M, M) array_like
        Schur transformation matrix
    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    T : (M, M) ndarray
        Complex Schur form of the original array
    Z : (M, M) ndarray
        Schur transformation matrix corresponding to the complex form

    See Also
    --------
    schur : Schur decomposition of an array

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import schur, rsf2csf
    >>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])
    >>> T, Z = schur(A)
    >>> T
    array([[ 2.65896708,  1.42440458, -1.92933439],
           [ 0.        , -0.32948354, -0.49063704],
           [ 0.        ,  1.31178921, -0.32948354]])
    >>> Z
    array([[0.72711591, -0.60156188, 0.33079564],
           [0.52839428, 0.79801892, 0.28976765],
           [0.43829436, 0.03590414, -0.89811411]])
    >>> T2 , Z2 = rsf2csf(T, Z)
    >>> T2
    array([[2.65896708+0.j, -1.64592781+0.743164187j, -1.21516887+1.00660462j],
           [0.+0.j , -0.32948354+8.02254558e-01j, -0.82115218-2.77555756e-17j],
           [0.+0.j , 0.+0.j, -0.32948354-0.802254558j]])
    >>> Z2
    array([[0.72711591+0.j,  0.28220393-0.31385693j,  0.51319638-0.17258824j],
           [0.52839428+0.j,  0.24720268+0.41635578j, -0.68079517-0.15118243j],
           [0.43829436+0.j, -0.76618703+0.01873251j, -0.03063006+0.46857912j]])

    """
    if check_finite:
        Z, T = map(asarray_chkfinite, (Z, T))
    else:
        Z, T = map(asarray, (Z, T))
    for ind, X in enumerate([Z, T]):
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise ValueError("Input '{}' must be square.".format('ZT'[ind]))
    if T.shape[0] != Z.shape[0]:
        message = f'Input array shapes must match: Z: {Z.shape} vs. T: {T.shape}'
        raise ValueError(message)
    N = T.shape[0]
    t = _commonType(Z, T, array([3.0], 'F'))
    Z, T = _castCopy(t, Z, T)
    for m in range(N - 1, 0, -1):
        if abs(T[m, m - 1]) > eps * (abs(T[m - 1, m - 1]) + abs(T[m, m])):
            mu = eigvals(T[m - 1:m + 1, m - 1:m + 1]) - T[m, m]
            r = norm([mu[0], T[m, m - 1]])
            c = mu[0] / r
            s = T[m, m - 1] / r
            G = array([[c.conj(), s], [-s, c]], dtype=t)
            T[m - 1:m + 1, m - 1:] = G.dot(T[m - 1:m + 1, m - 1:])
            T[:m + 1, m - 1:m + 1] = T[:m + 1, m - 1:m + 1].dot(G.conj().T)
            Z[:, m - 1:m + 1] = Z[:, m - 1:m + 1].dot(G.conj().T)
        T[m, m - 1] = 0.0
    return (T, Z)