import operator
from functools import reduce
from mmap import mmap
from numbers import Integral
import numpy as np
def slicers2segments(read_slicers, in_shape, offset, itemsize):
    """Get segments from `read_slicers` given `in_shape` and memory steps

    Parameters
    ----------
    read_slicers : object
        something that can be used to slice an array as in ``arr[sliceobj]``
        Slice objects can by be assumed canonical as in ``canonical_slicers``,
        and positive as in ``_positive_slice``
    in_shape : sequence
        shape of underlying array on disk before reading
    offset : int
        offset of array data in underlying file or memory buffer
    itemsize : int
        element size in array (in bytes)

    Returns
    -------
    segments : list
        list of 2 element lists where lists are [offset, length], giving
        absolute memory offset in bytes and number of bytes to read
    """
    all_full = True
    all_segments = [[offset, itemsize]]
    stride = itemsize
    real_no = 0
    for read_slicer in read_slicers:
        if read_slicer is None:
            continue
        dim_len = in_shape[real_no]
        real_no += 1
        is_int = isinstance(read_slicer, Integral)
        if not is_int:
            read_slicer = fill_slicer(read_slicer, dim_len)
            slice_len = _full_slicer_len(read_slicer)
        is_full = read_slicer == slice(0, dim_len, 1)
        is_contiguous = not is_int and read_slicer.step == 1
        if all_full and is_contiguous:
            if read_slicer.start != 0:
                all_segments[0][0] += stride * read_slicer.start
            all_segments[0][1] *= slice_len
        elif is_int:
            for segment in all_segments:
                segment[0] += stride * read_slicer
        else:
            segments = all_segments
            all_segments = []
            for i in range(read_slicer.start, read_slicer.stop, read_slicer.step):
                for s in segments:
                    all_segments.append([s[0] + stride * i, s[1]])
        all_full = all_full and is_full
        stride *= dim_len
    return all_segments