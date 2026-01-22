from itertools import groupby
from warnings import warn
import numpy as np
from scipy.sparse import find, coo_matrix
def validate_first_step(first_step, t0, t_bound):
    """Assert that first_step is valid and return it."""
    if first_step <= 0:
        raise ValueError('`first_step` must be positive.')
    if first_step > np.abs(t_bound - t0):
        raise ValueError('`first_step` exceeds bounds.')
    return first_step