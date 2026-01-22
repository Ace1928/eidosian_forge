import itertools
from typing import Union, Sequence, Optional
import numpy as np
from cirq.value import random_state
def in_weyl_chamber(kak_vec: np.ndarray) -> np.ndarray:
    """Whether a given collection of coordinates is within the Weyl chamber.

    Args:
        kak_vec: A numpy.ndarray tensor encoding a KAK 3-vector. Input may be
            broadcastable with shape (...,3).

    Returns:
        np.ndarray of boolean values denoting whether the given coordinates
        are in the Weyl chamber.
    """
    kak_vec = np.asarray(kak_vec)
    assert kak_vec.shape[-1] == 3, 'Last index of input must represent a 3-vector.'
    xp, yp, zp = (kak_vec[..., 0], kak_vec[..., 1], kak_vec[..., 2])
    pi_4 = np.pi / 4
    x_inside = np.logical_and(0 <= xp, xp <= pi_4)
    y_inside = np.logical_and(0 <= yp, yp <= pi_4)
    y_inside = np.logical_and(y_inside, xp >= yp)
    z_inside = np.abs(zp) <= yp
    return np.logical_and.reduce((x_inside, y_inside, z_inside))