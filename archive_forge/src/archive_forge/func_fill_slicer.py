import operator
from functools import reduce
from mmap import mmap
from numbers import Integral
import numpy as np
def fill_slicer(slicer, in_len):
    """Return slice object with Nones filled out to match `in_len`

    Also fixes too large stop / start values according to slice() slicing
    rules.

    The returned slicer can have a None as `slicer.stop` if `slicer.step` is
    negative and the input `slicer.stop` is None. This is because we can't
    represent the ``stop`` as an integer, because -1 has a different meaning.

    Parameters
    ----------
    slicer : slice object
    in_len : int
        length of axis on which `slicer` will be applied

    Returns
    -------
    can_slicer : slice object
        slice with start, stop, step set to explicit values, with the exception
        of ``stop`` for negative step, which is None for the case of slicing
        down through the first element
    """
    start, stop, step = (slicer.start, slicer.stop, slicer.step)
    if step is None:
        step = 1
    if start is not None and start < 0:
        start = in_len + start
    if stop is not None and stop < 0:
        stop = in_len + stop
    if step > 0:
        if start is None:
            start = 0
        if stop is None:
            stop = in_len
        else:
            stop = min(stop, in_len)
    elif start is None:
        start = in_len - 1
    else:
        start = min(start, in_len - 1)
    return slice(start, stop, step)