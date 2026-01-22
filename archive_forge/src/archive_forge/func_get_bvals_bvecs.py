import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from io import StringIO
from locale import getpreferredencoding
import numpy as np
from .affines import apply_affine, dot_reduce, from_matvec
from .eulerangles import euler2mat
from .fileslice import fileslice, strided_scalar
from .nifti1 import unit_codes
from .openers import ImageOpener
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import Recoder, array_from_file
def get_bvals_bvecs(self):
    """Get bvals and bvecs from data

        Returns
        -------
        b_vals : None or array
            Array of b values, shape (n_directions,), or None if not a
            diffusion acquisition.
        b_vectors : None or array
            Array of b vectors, shape (n_directions, 3), or None if not a
            diffusion acquisition.
        """
    if self.general_info['diffusion'] == 0:
        return (None, None)
    reorder = self.get_sorted_slice_indices()
    if len(self.get_data_shape()) == 3:
        return (None, None)
    else:
        n_slices, n_vols = self.get_data_shape()[-2:]
    bvals = self.image_defs['diffusion_b_factor'][reorder].reshape((n_slices, n_vols), order='F')
    assert not np.any(np.diff(bvals, axis=0))
    bvals = bvals[0]
    if 'diffusion' not in self.image_defs.dtype.names:
        return (bvals, None)
    bvecs = self.image_defs['diffusion'][reorder].reshape((n_slices, n_vols, 3), order='F')
    assert not np.any(np.diff(bvecs, axis=0))
    bvecs = bvecs[0]
    permute_to_psl = ACQ_TO_PSL[self.get_slice_orientation()]
    bvecs = apply_affine(np.linalg.inv(permute_to_psl), bvecs)
    return (bvals, bvecs)