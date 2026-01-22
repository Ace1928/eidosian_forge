from __future__ import division
import numbers
from typing import Optional, Tuple
import numpy as np
def none_to_empty(val):
    """Converts None to an empty string.
    """
    if val is None:
        return ''
    else:
        return val