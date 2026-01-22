import numpy as np
import scipy.linalg
def expm_frechet_block_enlarge(A, E):
    """
    This is a helper function, mostly for testing and profiling.
    Return expm(A), frechet(A, E)
    """
    n = A.shape[0]
    M = np.vstack([np.hstack([A, E]), np.hstack([np.zeros_like(A), A])])
    expm_M = scipy.linalg.expm(M)
    return (expm_M[:n, :n], expm_M[:n, n:])