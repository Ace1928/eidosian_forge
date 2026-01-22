from __future__ import annotations
from itertools import groupby
from math import floor, ceil
import dask.array as da
import numpy as np
from dask.delayed import delayed
from numba import prange
from .utils import ngjit, ngjit_parallel
def resample_2d_distributed(src, w, h, ds_method='mean', us_method='linear', fill_value=None, mode_rank=1, chunksize=None, max_mem=None):
    """
    A distributed version of 2-d grid resampling which operates on
    dask arrays and performs regridding on a chunked array.

    Parameters
    ----------
    src : dask.array.Array
        The source array to resample
    w : int
        New grid width
    h : int
        New grid height
    ds_method : str (optional)
        Grid cell aggregation method for a possible downsampling
        (one of the *DS_* constants).
    us_method : str (optional)
        Grid cell interpolation method for a possible upsampling
        (one of the *US_* constants, optional).
    fill_value : scalar (optional)
        If None, numpy's default value is used.
    mode_rank : scalar (optional)
        The rank of the frequency determined by the *ds_method*
        ``DS_MODE``. One (the default) means most frequent value, two
        means second most frequent value, and so forth.
    chunksize : tuple(int, int) (optional)
        Size of the output chunks. By default this the chunk size is
        inherited from the *src* array.
    max_mem : int (optional)
        The maximum number of bytes that should be loaded into memory
        during the regridding operation.

    Returns
    -------
    resampled : dask.array.Array
        A resampled version of the *src* array.
    """
    temp_chunks = compute_chunksize(src, w, h, chunksize, max_mem)
    if chunksize is None:
        chunksize = src.chunksize
    chunk_map = map_chunks(src.shape, (h, w), temp_chunks)
    out_chunks = {}
    for (i, j), chunk in chunk_map.items():
        inds = chunk['in']
        inx0, inx1 = inds['x']
        iny0, iny1 = inds['y']
        out = chunk['out']
        chunk_array = src[iny0:iny1, inx0:inx1]
        resampled = _resample_2d_delayed(chunk_array, out['w'], out['h'], ds_method, us_method, fill_value, mode_rank, inds['xoffset'], inds['yoffset'])
        out_chunks[i, j] = {'array': resampled, 'shape': (out['h'], out['w']), 'dtype': src.dtype, 'in': chunk['in'], 'out': out}
    rows = groupby(out_chunks.items(), lambda x: x[0][0])
    cols = []
    for i, row in rows:
        row = da.concatenate([da.from_delayed(chunk['array'], chunk['shape'], chunk['dtype']) for _, chunk in row], 1)
        cols.append(row)
    out = da.concatenate(cols, 0)
    if chunksize is not None and out.chunksize != chunksize:
        out = out.rechunk(chunksize)
    return out