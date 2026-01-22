import numpy as np
from scipy.special import factorial
def li4(z):
    """Polylogarithm for negative integer order -4

    Li(-4, z)
    """
    return z * (1 + z) * (1 + 10 * z + z ** 2) / (1 - z) ** 5