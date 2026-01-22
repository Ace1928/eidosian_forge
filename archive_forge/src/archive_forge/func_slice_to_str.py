from __future__ import division
import numbers
from typing import Optional, Tuple
import numpy as np
def slice_to_str(slc):
    """Converts a slice into a string.
    """
    if is_single_index(slc):
        return str(slc.start)
    endpoints = [none_to_empty(val) for val in (slc.start, slc.stop)]
    if slc.step is not None and slc.step != 1:
        return '%s:%s:%s' % (endpoints[0], endpoints[1], slc.step)
    else:
        return '%s:%s' % (endpoints[0], endpoints[1])