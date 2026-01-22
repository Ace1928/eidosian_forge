import cupy
from cupyx.scipy.signal import windows
def cfar_alpha(pfa, N):
    """
    Computes the value of alpha corresponding to a given probability
    of false alarm and number of reference cells N.

    Parameters
    ----------
    pfa : float
        Probability of false alarm.

    N : int
        Number of reference cells.

    Returns
    -------
    alpha : float
        Alpha value.
    """
    return N * (pfa ** (-1.0 / N) - 1)