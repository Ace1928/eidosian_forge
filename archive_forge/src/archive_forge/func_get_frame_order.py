import warnings
from numbers import Integral
import numpy as np
from .arraywriters import make_array_writer
from .fileslice import canonical_slicers, predict_shape, slice2outax
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import array_from_file, make_dt_codes, native_code, swapped_code
from .wrapstruct import WrapStruct
def get_frame_order(mlist):
    """Returns the order of the frames stored in the file
    Sometimes Frames are not stored in the file in
    chronological order, this can be used to extract frames
    in correct order

    Returns
    -------
    id_dict: dict mapping frame number -> [mlist_row, mlist_id]

    (where mlist id is value in the first column of the mlist matrix )

    Examples
    --------
    >>> import os
    >>> import nibabel as nib
    >>> nibabel_dir = os.path.dirname(nib.__file__)
    >>> from nibabel import ecat
    >>> ecat_file = os.path.join(nibabel_dir,'tests','data','tinypet.v')
    >>> img = ecat.load(ecat_file)
    >>> mlist = img.get_mlist()
    >>> get_frame_order(mlist)
    {0: [0, 16842758]}
    """
    ids = mlist[:, 0].copy()
    n_valid = np.sum(ids > 0)
    ids[ids <= 0] = ids.max() + 1
    valid_order = np.argsort(ids)
    if not all(valid_order == sorted(valid_order)):
        warnings.warn_explicit(f'Frames stored out of order; true order = {valid_order}\nframes will be accessed in order STORED, NOT true order', UserWarning, 'ecat', 0)
    id_dict = {}
    for i in range(n_valid):
        id_dict[i] = [valid_order[i], ids[valid_order[i]]]
    return id_dict