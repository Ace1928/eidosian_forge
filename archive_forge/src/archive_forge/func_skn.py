import functools
import warnings
import numpy as np
from ase.utils import IOContext
def skn(s, k, n):
    """Convert k or (s, k) to string."""
    if kpts is None:
        return '(s={}, k={}, n={})'.format(s, k, n)
    return '(s={}, k={}, n={}, [{:.2f}, {:.2f}, {:.2f}])'.format(s, k, n, *kpts[k])