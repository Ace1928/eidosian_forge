import numpy as np
import nibabel as nib
from nibabel.streamlines import FORMATS
from nibabel.streamlines.header import Field
Marks every nonzero voxel using streamlines to form a 3D 'X' inside.

    Generates streamlines forming a 3D 'X' inside every nonzero voxel.

    Parameters
    ----------
    mask : ndarray
        Mask containing the spots to be marked.

    Returns
    -------
    list of ndarrays
        All streamlines needed to mark every nonzero voxel in the `mask`.
    