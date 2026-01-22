import warnings
from .. import h5ds
from ..h5py_warnings import H5pyDeprecationWarning
from . import base
from .base import phil, with_phil
from .dataset import Dataset
def detach_scale(self, dset):
    """ Remove a scale from this dimension.

        Provide the Dataset of the scale you would like to remove.
        """
    with phil:
        h5ds.detach_scale(self._id, dset.id, self._dimension)