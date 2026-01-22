import numpy as np
import numpy.linalg as npl
def q2bg(q_vector, tol=1e-05):
    """Return b value and q unit vector from q vector `q_vector`

    Parameters
    ----------
    q_vector : (3,) array-like
        q vector
    tol : float, optional
        q vector L2 norm below which `q_vector` considered to be `b_value` of
        zero, and therefore `g_vector` also considered to zero.

    Returns
    -------
    b_value : float
        L2 Norm of `q_vector` or 0 if L2 norm < `tol`
    g_vector : shape (3,) ndarray
        `q_vector` / `b_value` or 0 if L2 norma < `tol`

    Examples
    --------
    >>> q2bg([1, 0, 0])
    (1.0, array([1., 0., 0.]))
    >>> q2bg([0, 10, 0])
    (10.0, array([0., 1., 0.]))
    >>> q2bg([0, 0, 0])
    (0.0, array([0., 0., 0.]))
    """
    q_vec = np.asarray(q_vector)
    norm = np.sqrt(np.sum(q_vec * q_vec))
    if norm < tol:
        return (0.0, np.zeros((3,)))
    return (norm, q_vec / norm)