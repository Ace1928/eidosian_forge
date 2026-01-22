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
@with_phil
def iter_chunks(self, sel=None):
    """ Return chunk iterator.  If set, the sel argument is a slice or
        tuple of slices that defines the region to be used. If not set, the
        entire dataspace will be used for the iterator.

        For each chunk within the given region, the iterator yields a tuple of
        slices that gives the intersection of the given chunk with the
        selection area.

        A TypeError will be raised if the dataset is not chunked.

        A ValueError will be raised if the selection region is invalid.

        """
    return ChunkIterator(self, sel)