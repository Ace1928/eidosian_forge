from __future__ import annotations
import numba as nb
import numpy as np
import os
@nb.jit('(float64, float64, float64, float64)', nopython=True, nogil=True, cache=True)
def combine_scaled(r, g, b, a):
    """Combine components in [0, 1] to rgba uint32"""
    r2 = min(255, np.uint32(r * 255))
    g2 = min(255, np.uint32(g * 255))
    b2 = min(255, np.uint32(b * 255))
    a2 = min(255, np.uint32(a * 255))
    return np.uint32(a2 << 24 | b2 << 16 | g2 << 8 | r2)