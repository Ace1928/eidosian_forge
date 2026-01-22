import fnmatch
import locale
import os
import re
import stat
import subprocess
import sys
import textwrap
import types
import warnings
from xml.etree import ElementTree
def slice_bounds(sequence, slice_obj, allow_step=False):
    """
    Given a slice, return the corresponding (start, stop) bounds,
    taking into account None indices and negative indices.  The
    following guarantees are made for the returned start and stop values:

      - 0 <= start <= len(sequence)
      - 0 <= stop <= len(sequence)
      - start <= stop

    :raise ValueError: If ``slice_obj.step`` is not None.
    :param allow_step: If true, then the slice object may have a
        non-None step.  If it does, then return a tuple
        (start, stop, step).
    """
    start, stop = (slice_obj.start, slice_obj.stop)
    if allow_step:
        step = slice_obj.step
        if step is None:
            step = 1
        if step < 0:
            start, stop = slice_bounds(sequence, slice(stop, start))
        else:
            start, stop = slice_bounds(sequence, slice(start, stop))
        return (start, stop, step)
    elif slice_obj.step not in (None, 1):
        raise ValueError('slices with steps are not supported by %s' % sequence.__class__.__name__)
    if start is None:
        start = 0
    if stop is None:
        stop = len(sequence)
    if start < 0:
        start = max(0, len(sequence) + start)
    if stop < 0:
        stop = max(0, len(sequence) + stop)
    if stop > 0:
        try:
            sequence[stop - 1]
        except IndexError:
            stop = len(sequence)
    start = min(start, stop)
    return (start, stop)