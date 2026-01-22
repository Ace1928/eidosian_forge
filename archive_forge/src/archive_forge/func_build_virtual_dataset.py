from contextlib import contextmanager
import posixpath as pp
import numpy
from .compat import filename_decode, filename_encode
from .. import h5, h5g, h5i, h5o, h5r, h5t, h5l, h5p
from . import base
from .base import HLObject, MutableMappingHDF5, phil, with_phil
from . import dataset
from . import datatype
from .vds import vds_support
@contextmanager
def build_virtual_dataset(self, name, shape, dtype, maxshape=None, fillvalue=None):
    """Assemble a virtual dataset in this group.

            This is used as a context manager::

                with f.build_virtual_dataset('virt', (10, 1000), np.uint32) as layout:
                    layout[0] = h5py.VirtualSource('foo.h5', 'data', (1000,))

            name
                (str) Name of the new dataset
            shape
                (tuple) Shape of the dataset
            dtype
                A numpy dtype for data read from the virtual dataset
            maxshape
                (tuple, optional) Maximum dimensions if the dataset can grow.
                Use None for unlimited dimensions.
            fillvalue
                The value used where no data is available.
            """
    from .vds import VirtualLayout
    layout = VirtualLayout(shape, dtype, maxshape, self.file.filename)
    yield layout
    self.create_virtual_dataset(name, layout, fillvalue)