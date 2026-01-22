import numpy as np
import scipy as sp
def promax(A, k=2):
    """
    Performs promax rotation of the matrix :math:`A`.

    This method was not very clear to me from the literature, this
    implementation is as I understand it should work.

    Promax rotation is performed in the following steps:

    * Determine varimax rotated patterns :math:`V`.

    * Construct a rotation target matrix :math:`|V_{ij}|^k/V_{ij}`

    * Perform procrustes rotation towards the target to obtain T

    * Determine the patterns

    First, varimax rotation a target matrix :math:`H` is determined with
    orthogonal varimax rotation.
    Then, oblique target rotation is performed towards the target.

    Parameters
    ----------
    A : numpy matrix
        non rotated factors
    k : float
        parameter, should be positive

    References
    ----------
    [1] Browne (2001) - An overview of analytic rotation in exploratory
    factor analysis

    [2] Navarra, Simoncini (2010) - A guide to empirical orthogonal functions
    for climate data analysis
    """
    assert k > 0
    from ._wrappers import rotate_factors
    V, T = rotate_factors(A, 'varimax')
    H = np.abs(V) ** k / V
    S = procrustes(A, H)
    d = np.sqrt(np.diag(np.linalg.inv(S.T.dot(S))))
    D = np.diag(d)
    T = np.linalg.inv(S.dot(D)).T
    return (A.dot(T), T)