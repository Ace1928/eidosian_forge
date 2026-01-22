from collections.abc import (
import os
import posixpath
import numpy as np
from .._objects import phil, with_phil
from .. import h5d, h5i, h5r, h5p, h5f, h5t, h5s
from .compat import fspath, filename_encode
def selection(self, ref):
    """ Get the shape of the target dataspace selection referred to by *ref*
        """
    from . import selections
    with phil:
        sid = h5r.get_region(ref, self.id)
        return selections.guess_shape(sid)