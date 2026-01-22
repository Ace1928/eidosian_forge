from __future__ import annotations
import warnings
from io import BytesIO
import numpy as np
import numpy.linalg as npl
from . import analyze  # module import
from .arrayproxy import get_obj_dtype
from .batteryrunners import Report
from .casting import have_binary128
from .deprecated import alert_future_error
from .filebasedimages import ImageFileError, SerializableImage
from .optpkg import optional_package
from .quaternions import fillpositive, mat2quat, quat2mat
from .spatialimages import HeaderDataError
from .spm99analyze import SpmAnalyzeHeader
from .volumeutils import Recoder, endian_codes, make_dt_codes
def get_slice_times(self):
    """Get slice times from slice timing information

        Returns
        -------
        slice_times : tuple
            Times of acquisition of slices, where 0 is the beginning of
            the acquisition, ordered by position in file.  nifti allows
            slices at the top and bottom of the volume to be excluded from
            the standard slice timing specification, and calls these
            "padding slices".  We give padding slices ``None`` as a time
            of acquisition

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_dim_info(slice=2)
        >>> hdr.set_data_shape((1, 1, 7))
        >>> hdr.set_slice_duration(0.1)
        >>> hdr['slice_code'] = slice_order_codes['sequential increasing']
        >>> slice_times = hdr.get_slice_times()
        >>> np.allclose(slice_times, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        True
        """
    hdr = self._structarr
    slice_len = self.get_n_slices()
    duration = self.get_slice_duration()
    slabel = self.get_value_label('slice_code')
    if slabel == 'unknown':
        raise HeaderDataError('Cannot get slice times when slice code is "unknown"')
    slice_start, slice_end = (int(hdr['slice_start']), int(hdr['slice_end']))
    if slice_start < 0:
        raise HeaderDataError('slice_start should be >= 0')
    if slice_end == 0:
        slice_end = slice_len - 1
    n_timed = slice_end - slice_start + 1
    if n_timed < 1:
        raise HeaderDataError('slice_end should be > slice_start')
    st_order = self._slice_time_order(slabel, n_timed)
    times = st_order * duration
    return (None,) * slice_start + tuple(times) + (None,) * (slice_len - slice_end - 1)