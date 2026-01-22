from itertools import product
import numpy as np
from .affines import apply_affine
def slice2volume(index, axis, shape=None):
    """Affine expressing selection of a single slice from 3D volume

    Imagine we have taken a slice from an image data array, ``s = data[:, :,
    index]``.  This function returns the affine to map the array coordinates of
    ``s`` to the array coordinates of ``data``.

    This can be useful for resampling a single slice from a volume.  For
    example, to resample slice ``k`` in the space of ``img1`` from the matching
    spatial voxel values in ``img2``, you might do something like::

        slice_shape = img1.shape[:2]
        slice_aff = slice2volume(k, 2)
        whole_aff = np.linalg.inv(img2.affine).dot(img1.affine.dot(slice_aff))

    and then use ``whole_aff`` in ``scipy.ndimage.affine_transform``:

        rzs, trans = to_matvec(whole_aff)
        data = img2.get_fdata()
        new_slice = scipy.ndimage.affine_transform(data, rzs, trans, slice_shape)

    Parameters
    ----------
    index : int
        index of selected slice
    axis : {0, 1, 2}
        axis to which `index` applies

    Returns
    -------
    slice_aff : shape (4, 3) affine
        Affine relating input coordinates in a slice to output coordinates in
        the embedded volume
    """
    if index < 0:
        raise ValueError('Cannot handle negative index')
    if not 0 <= axis <= 2:
        raise ValueError('Axis should be between 0 and 2')
    axes = list(range(4))
    axes.remove(axis)
    slice_aff = np.eye(4)[:, axes]
    slice_aff[axis, -1] = index
    return slice_aff