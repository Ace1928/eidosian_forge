import numpy as np
def forrt(X, m=None):
    """
    RFFT with order like Munro (1976) FORTT routine.
    """
    if m is None:
        m = len(X)
    y = np.fft.rfft(X, m) / m
    return np.r_[y.real, y[1:-1].imag]