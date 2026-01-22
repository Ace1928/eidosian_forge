import numpy as np
from scipy.special import factorial
def li3(z):
    """Polylogarithm for negative integer order -3

    Li(-3, z)
    """
    return z * (1 + 4 * z + z ** 2) / (1 - z) ** 4