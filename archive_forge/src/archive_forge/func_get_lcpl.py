from collections.abc import (
import os
import posixpath
import numpy as np
from .._objects import phil, with_phil
from .. import h5d, h5i, h5r, h5p, h5f, h5t, h5s
from .compat import fspath, filename_encode
def get_lcpl(coding):
    """ Create an appropriate link creation property list """
    lcpl = self._lcpl.copy()
    lcpl.set_char_encoding(coding)
    return lcpl