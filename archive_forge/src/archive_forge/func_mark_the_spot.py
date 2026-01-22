import numpy as np
import nibabel as nib
from nibabel.streamlines import FORMATS
from nibabel.streamlines.header import Field
def mark_the_spot(mask):
    """Marks every nonzero voxel using streamlines to form a 3D 'X' inside.

    Generates streamlines forming a 3D 'X' inside every nonzero voxel.

    Parameters
    ----------
    mask : ndarray
        Mask containing the spots to be marked.

    Returns
    -------
    list of ndarrays
        All streamlines needed to mark every nonzero voxel in the `mask`.
    """

    def _gen_straight_streamline(start, end, steps=3):
        coords = []
        for s, e in zip(start, end):
            coords.append(np.linspace(s, e, steps))
        return np.array(coords).T
    X = [_gen_straight_streamline((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)), _gen_straight_streamline((-0.5, 0.5, -0.5), (0.5, -0.5, 0.5)), _gen_straight_streamline((-0.5, 0.5, 0.5), (0.5, -0.5, -0.5)), _gen_straight_streamline((-0.5, -0.5, 0.5), (0.5, 0.5, -0.5))]
    coords = np.array(zip(*np.where(mask)))
    streamlines = [(line + c) * voxel_size for c in coords for line in X]
    return streamlines