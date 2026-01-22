from __future__ import division
import numbers
from typing import Optional, Tuple
import numpy as np
def format_slice(key_val, dim, axis) -> Optional[slice]:
    """Converts part of a key into a slice with a start and step.

    Uses the same syntax as numpy.

    Args:
        key_val: The value to convert into a slice.
        dim: The length of the dimension being sliced.

    Returns:
        A slice with a start and step.
    """
    if key_val is None:
        return None
    elif isinstance(key_val, slice):
        step = to_int(key_val.step, 1)
        if step == 0:
            raise ValueError('step length cannot be 0')
        elif step > 0:
            start = np.clip(wrap_neg_index(to_int(key_val.start, 0), dim), 0, dim)
            stop = np.clip(wrap_neg_index(to_int(key_val.stop, dim), dim), 0, dim)
        else:
            start = np.clip(wrap_neg_index(to_int(key_val.start, dim - 1), dim), -1, dim - 1)
            stop = np.clip(wrap_neg_index(to_int(key_val.stop, -dim - 1), dim, True), -1, dim - 1)
        return slice(start, stop, step)
    else:
        orig_key_val = to_int(key_val)
        key_val = wrap_neg_index(orig_key_val, dim)
        if 0 <= key_val < dim:
            return slice(key_val, key_val + 1, 1)
        else:
            raise IndexError('Index %i is out of bounds for axis %i with size %i.' % (orig_key_val, axis, dim))