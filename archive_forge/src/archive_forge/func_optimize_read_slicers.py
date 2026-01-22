import operator
from functools import reduce
from mmap import mmap
from numbers import Integral
import numpy as np
def optimize_read_slicers(sliceobj, in_shape, itemsize, heuristic):
    """Calculates slices to read from disk, and apply after reading

    Parameters
    ----------
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``.
        Can be assumed to be canonical in the sense of ``canonical_slicers``
    in_shape : sequence
        shape of underlying array to be sliced.  Array for `in_shape` assumed
        to be already in 'F' order. Reorder shape / sliceobj for slicing a 'C'
        array before passing to this function.
    itemsize : int
        element size in array (bytes)
    heuristic : callable
        function taking slice object, axis length, and stride length as
        arguments, returning one of 'full', 'contiguous', None.  See
        :func:`optimize_slicer`; see :func:`threshold_heuristic` for an
        example.

    Returns
    -------
    read_slicers : tuple
        `sliceobj` maybe rephrased to fill out dimensions that are better read
        from disk and later trimmed to their original size with `post_slicers`.
        `read_slicers` implies a block of memory to be read from disk. The
        actual disk positions come from `slicers2segments` run over
        `read_slicers`. Includes any ``newaxis`` dimensions in `sliceobj`
    post_slicers : tuple
        Any new slicing to be applied to the read array after reading.  The
        `post_slicers` discard any memory that we read to save time, but that
        we don't need for the slice.  Include any ``newaxis`` dimension added
        by `sliceobj`
    """
    read_slicers = []
    post_slicers = []
    real_no = 0
    stride = itemsize
    all_full = True
    for slicer in sliceobj:
        if slicer is None:
            read_slicers.append(None)
            post_slicers.append(slice(None))
            continue
        dim_len = in_shape[real_no]
        real_no += 1
        is_last = real_no == len(in_shape)
        read_slicer, post_slicer = optimize_slicer(slicer, dim_len, all_full, is_last, stride, heuristic)
        read_slicers.append(read_slicer)
        all_full = all_full and read_slicer == slice(None)
        if not isinstance(read_slicer, Integral):
            post_slicers.append(post_slicer)
        stride *= dim_len
    return (tuple(read_slicers), tuple(post_slicers))