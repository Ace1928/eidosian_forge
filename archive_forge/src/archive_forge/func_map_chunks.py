from __future__ import annotations
from itertools import groupby
from math import floor, ceil
import dask.array as da
import numpy as np
from dask.delayed import delayed
from numba import prange
from .utils import ngjit, ngjit_parallel
def map_chunks(in_shape, out_shape, out_chunks):
    """
    Maps index in source array to target array chunks.

    For each chunk in the target array this function computes the
    indexes into the source array that will be fed into the regridding
    operation.

    Parameters
    ----------
    in_shape: tuple(int, int)
      The shape of the input array
    out_shape: tuple(int, int)
      The shape of the output array
    out_chunks: tuple(int, int)
      The shape of each chunk in the output array

    Returns
    -------
      Dictionary mapping of chunks and their indexes
      in the input and output array.
    """
    outy, outx = out_shape
    cys, cxs = out_chunks
    xchunks = list(range(0, outx, cxs)) + [outx]
    ychunks = list(range(0, outy, cys)) + [outy]
    iny, inx = in_shape
    xscale = inx / outx
    yscale = iny / outy
    mapping = {}
    for i in range(len(ychunks) - 1):
        cumy0, cumy1 = ychunks[i:i + 2]
        iny0, iny1 = (cumy0 * yscale, cumy1 * yscale)
        iny0r, iny1r = (floor(iny0), ceil(iny1))
        y0_off, y1_off = (iny0 - iny0r, iny1r - iny1)
        for j in range(len(xchunks) - 1):
            cumx0, cumx1 = xchunks[j:j + 2]
            inx0, inx1 = (cumx0 * xscale, cumx1 * xscale)
            inx0r, inx1r = (floor(inx0), ceil(inx1))
            x0_off, x1_off = (inx0 - inx0r, inx1r - inx1)
            mapping[i, j] = {'out': {'x': (cumx0, cumx1), 'y': (cumy0, cumy1), 'w': cumx1 - cumx0, 'h': cumy1 - cumy0}, 'in': {'x': (inx0r, inx1r), 'y': (iny0r, iny1r), 'xoffset': (x0_off, x1_off), 'yoffset': (y0_off, y1_off)}}
    return mapping