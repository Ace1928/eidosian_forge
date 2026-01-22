from collections.abc import (
import os
import posixpath
import numpy as np
from .._objects import phil, with_phil
from .. import h5d, h5i, h5r, h5p, h5f, h5t, h5s
from .compat import fspath, filename_encode
def is_empty_dataspace(obj):
    """ Check if an object's dataspace is empty """
    if obj.get_space().get_simple_extent_type() == h5s.NULL:
        return True
    return False