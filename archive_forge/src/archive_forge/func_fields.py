import posixpath as pp
import sys
import numpy
from .. import h5, h5s, h5t, h5r, h5d, h5p, h5fd, h5ds, _selector
from .base import (
from . import filters
from . import selections as sel
from . import selections2 as sel2
from .datatype import Datatype
from .compat import filename_decode
from .vds import VDSmap, vds_support
def fields(self, names, *, _prior_dtype=None):
    """Get a wrapper to read a subset of fields from a compound data type:

        >>> 2d_coords = dataset.fields(['x', 'y'])[:]

        If names is a string, a single field is extracted, and the resulting
        arrays will have that dtype. Otherwise, it should be an iterable,
        and the read data will have a compound dtype.
        """
    if _prior_dtype is None:
        _prior_dtype = self.dtype
    return FieldsWrapper(self, _prior_dtype, names)