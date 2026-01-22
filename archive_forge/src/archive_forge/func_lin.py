import numpy as np
from scipy.special import factorial
def lin(n, z):
    """Polylogarithm for negative integer order -n

    Li(-n, z)

    https://en.wikipedia.org/wiki/Polylogarithm#Particular_values
    """
    if np.size(z) > 1:
        z = np.array(z)[..., None]
    k = np.arange(n + 1)
    st2 = np.array([sterling2(n + 1, ki + 1) for ki in k])
    res = (-1) ** (n + 1) * np.sum(factorial(k) * st2 * (-1 / (1 - z)) ** (k + 1), axis=-1)
    return res