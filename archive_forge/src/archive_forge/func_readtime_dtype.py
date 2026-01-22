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
def readtime_dtype(basetype, names):
    """Make a NumPy compound dtype with a subset of available fields"""
    if basetype.names is None:
        raise ValueError('Field names only allowed for compound types')
    for name in names:
        if name not in basetype.names:
            raise ValueError('Field %s does not appear in this type.' % name)
    return numpy.dtype([(name, basetype.fields[name][0]) for name in names])