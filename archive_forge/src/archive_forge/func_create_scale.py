import warnings
from .. import h5ds
from ..h5py_warnings import H5pyDeprecationWarning
from . import base
from .base import phil, with_phil
from .dataset import Dataset
def create_scale(self, dset, name=''):
    """ Create a new dimension, from an initial scale.

        Provide the dataset and a name for the scale.
        """
    warnings.warn('other_ds.dims.create_scale(ds, name) is deprecated. Use ds.make_scale(name) instead.', H5pyDeprecationWarning, stacklevel=2)
    dset.make_scale(name)