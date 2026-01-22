import weakref
import numpy as np
from .affines import voxel_sizes
from .optpkg import optional_package
from .orientations import aff2axcodes, axcodes2ornt
@property
def n_volumes(self):
    """Number of volumes in the data"""
    return int(np.prod(self._volume_dims))