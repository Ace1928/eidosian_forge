import numpy as np
from scipy import sparse
from pygsp import utils
def cheby_rect(G, bounds, signal, **kwargs):
    """
    Fast filtering using Chebyshev polynomial for a perfect rectangle filter.

    Parameters
    ----------
    G : Graph
    bounds : array-like
        The bounds of the pass-band filter
    signal : array-like
        Signal to filter
    order : int (optional)
        Order of the Chebyshev polynomial (default: 30)

    Returns
    -------
    r : array-like
        Result of the filtering

    """
    if not (isinstance(bounds, (list, np.ndarray)) and len(bounds) == 2):
        raise ValueError('Bounds of wrong shape.')
    bounds = np.array(bounds)
    m = int(kwargs.pop('order', 30) + 1)
    try:
        Nv = np.shape(signal)[1]
        r = np.zeros((G.N, Nv))
    except IndexError:
        r = np.zeros(G.N)
    b1, b2 = np.arccos(2.0 * bounds / G.lmax - 1.0)
    factor = 4.0 / G.lmax * G.L - 2.0 * sparse.eye(G.N)
    T_old = signal
    T_cur = factor.dot(signal) / 2.0
    r = (b1 - b2) / np.pi * signal + 2.0 / np.pi * (np.sin(b1) - np.sin(b2)) * T_cur
    for k in range(2, m):
        T_new = factor.dot(T_cur) - T_old
        r += 2.0 / (k * np.pi) * (np.sin(k * b1) - np.sin(k * b2)) * T_new
        T_old = T_cur
        T_cur = T_new
    return r