from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Optional
import numpy as np
from .fast_grad_utils import (
def init_layer1q_deriv_matrices(thetas: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Initializes 4x4 derivative matrices of 2-qubit gates defined in the paper.

    Args:
        thetas: depth x 4 matrix of gate parameters for every layer, where
                "depth" is the number of layers.
        dst: destination array of size depth x 4 x 4 x 4 that will receive gate
             derivative matrices of each layer; there are 4 parameters per gate,
             hence, 4 derivative matrices per layer.

    Returns:
        Returns the "dst" array.
    """
    n = thetas.shape[0]
    y = np.asarray([[0, -0.5], [0.5, 0]], dtype=np.complex128)
    z = np.asarray([[-0.5j, 0], [0, 0.5j]], dtype=np.complex128)
    tmp = np.full((5, 2, 2), fill_value=0, dtype=np.complex128)
    for k in range(n):
        th = thetas[k]
        a = make_rz(th[0], out=tmp[0])
        b = make_ry(th[1], out=tmp[1])
        c = make_rz(th[2], out=tmp[2])
        za = np.dot(z, a, out=tmp[3])
        np.dot(np.dot(za, b, out=tmp[4]), c, out=dst[k, 0])
        yb = np.dot(y, b, out=tmp[3])
        np.dot(a, np.dot(yb, c, out=tmp[4]), out=dst[k, 1])
        zc = np.dot(z, c, out=tmp[3])
        np.dot(a, np.dot(b, zc, out=tmp[4]), out=dst[k, 2])
    return dst