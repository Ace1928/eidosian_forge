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
def make_scale(self, name=''):
    """Make this dataset an HDF5 dimension scale.

        You can then attach it to dimensions of other datasets like this::

            other_ds.dims[0].attach_scale(ds)

        You can optionally pass a name to associate with this scale.
        """
    h5ds.set_scale(self._id, self._e(name))