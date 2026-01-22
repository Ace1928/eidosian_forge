import operator
from functools import reduce
from mmap import mmap
from numbers import Integral
import numpy as np
def threshold_heuristic(slicer, dim_len, stride, skip_thresh=SKIP_THRESH):
    """Whether to force full axis read or contiguous read of stepped slice

    Allows :func:`fileslice` to sometimes read memory that it will throw away
    in order to get maximum speed.  In other words, trade memory for fewer disk
    reads.

    Parameters
    ----------
    slicer : slice object, or int
        If slice, can be assumed to be full as in ``fill_slicer``
    dim_len : int
        length of axis being sliced
    stride : int
        memory distance between elements on this axis
    skip_thresh : int, optional
        Memory gap threshold in bytes above which to prefer skipping memory
        rather than reading it and later discarding.

    Returns
    -------
    action : {'full', 'contiguous', None}
        Gives the suggested optimization for reading the data

        * 'full' - read whole axis
        * 'contiguous' - read all elements between start and stop
        * None - read only memory needed for output

    Notes
    -----
    Let's say we are in the middle of reading a file at the start of some
    memory length $B$ bytes.  We don't need the memory, and we are considering
    whether to read it anyway (then throw it away) (READ) or stop reading, skip
    $B$ bytes and restart reading from there (SKIP).

    After trying some more fancy algorithms, a hard threshold (`skip_thresh`)
    for the maximum skip distance seemed to work well, as measured by times on
    ``nibabel.benchmarks.bench_fileslice``
    """
    if isinstance(slicer, Integral):
        gap_size = (dim_len - 1) * stride
        return 'full' if gap_size <= skip_thresh else None
    step_size = abs(slicer.step) * stride
    if step_size > skip_thresh:
        return None
    slicer = _positive_slice(slicer)
    start, stop = (slicer.start, slicer.stop)
    read_len = stop - start
    gap_size = (dim_len - read_len) * stride
    return 'full' if gap_size <= skip_thresh else 'contiguous'