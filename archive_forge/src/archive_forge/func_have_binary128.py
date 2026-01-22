from __future__ import annotations
import warnings
from platform import machine, processor
import numpy as np
from .deprecated import deprecate_with_version
def have_binary128():
    """True if we have a binary128 IEEE longdouble"""
    try:
        ti = type_info(np.longdouble)
    except FloatingError:
        return False
    return (ti['nmant'], ti['maxexp']) == (112, 16384)