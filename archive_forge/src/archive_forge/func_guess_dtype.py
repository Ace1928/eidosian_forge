from collections.abc import (
import os
import posixpath
import numpy as np
from .._objects import phil, with_phil
from .. import h5d, h5i, h5r, h5p, h5f, h5t, h5s
from .compat import fspath, filename_encode
def guess_dtype(data):
    """ Attempt to guess an appropriate dtype for the object, returning None
    if nothing is appropriate (or if it should be left up the the array
    constructor to figure out)
    """
    with phil:
        if isinstance(data, h5r.RegionReference):
            return h5t.regionref_dtype
        if isinstance(data, h5r.Reference):
            return h5t.ref_dtype
        item_type = find_item_type(data)
        if item_type is bytes:
            return h5t.string_dtype(encoding='ascii')
        if item_type is str:
            return h5t.string_dtype()
        return None