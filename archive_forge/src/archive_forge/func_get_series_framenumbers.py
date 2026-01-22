import warnings
from numbers import Integral
import numpy as np
from .arraywriters import make_array_writer
from .fileslice import canonical_slicers, predict_shape, slice2outax
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import array_from_file, make_dt_codes, native_code, swapped_code
from .wrapstruct import WrapStruct
def get_series_framenumbers(mlist):
    """Returns framenumber of data as it was collected,
    as part of a series; not just the order of how it was
    stored in this or across other files

    For example, if the data is split between multiple files
    this should give you the true location of this frame as
    collected in the series
    (Frames are numbered starting at ONE (1) not Zero)

    Returns
    -------
    frame_dict: dict mapping order_stored -> frame in series
            where frame in series counts from 1; [1,2,3,4...]

    Examples
    --------
    >>> import os
    >>> import nibabel as nib
    >>> nibabel_dir = os.path.dirname(nib.__file__)
    >>> from nibabel import ecat
    >>> ecat_file = os.path.join(nibabel_dir,'tests','data','tinypet.v')
    >>> img = ecat.load(ecat_file)
    >>> mlist = img.get_mlist()
    >>> get_series_framenumbers(mlist)
    {0: 1}
    """
    nframes = len(mlist)
    frames_order = get_frame_order(mlist)
    mlist_nframes = len(frames_order)
    trueframenumbers = np.arange(nframes - mlist_nframes, nframes)
    frame_dict = {}
    for frame_stored, (true_order, _) in frames_order.items():
        try:
            frame_dict[frame_stored] = trueframenumbers[true_order] + 1
        except IndexError:
            raise OSError('Error in header or mlist order unknown')
    return frame_dict