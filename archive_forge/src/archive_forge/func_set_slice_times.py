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
def set_slice_times(self, slice_times):
    """Set slice times into *hdr*

        Parameters
        ----------
        slice_times : tuple
            tuple of slice times, one value per slice
            tuple can include None to indicate no slice time for that slice

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_dim_info(slice=2)
        >>> hdr.set_data_shape([1, 1, 7])
        >>> hdr.set_slice_duration(0.1)
        >>> times = [None, 0.2, 0.4, 0.1, 0.3, 0.0, None]
        >>> hdr.set_slice_times(times)
        >>> hdr.get_value_label('slice_code')
        'alternating decreasing'
        >>> int(hdr['slice_start'])
        1
        >>> int(hdr['slice_end'])
        5
        """
    hdr = self._structarr
    slice_len = self.get_n_slices()
    if slice_len != len(slice_times):
        raise HeaderDataError('Number of slice times does not match number of slices')
    for ind, time in enumerate(slice_times):
        if time is not None:
            slice_start = ind
            break
    else:
        raise HeaderDataError('Not all slice times can be None')
    for ind, time in enumerate(slice_times[::-1]):
        if time is not None:
            slice_end = slice_len - ind - 1
            break
    timed = slice_times[slice_start:slice_end + 1]
    for time in timed:
        if time is None:
            raise HeaderDataError('Cannot have None in middle of slice time vector')
    tdiffs = np.diff(np.sort(timed))
    if not np.allclose(np.diff(tdiffs), 0):
        raise HeaderDataError('Slice times not compatible with single slice duration')
    duration = np.mean(tdiffs)
    st_order = np.round(np.array(timed) / duration)
    n_timed = len(timed)
    so_recoder = self._field_recoders['slice_code']
    labels = so_recoder.value_set('label')
    labels.remove('unknown')
    matching_labels = [label for label in labels if np.all(st_order == self._slice_time_order(label, n_timed))]
    if not matching_labels:
        raise HeaderDataError(f'slice ordering of {st_order} fits with no known scheme')
    if len(matching_labels) > 1:
        warnings.warn(f'Multiple slice orders satisfy: {', '.join(matching_labels)}. Choosing the first one')
    label = matching_labels[0]
    hdr['slice_start'] = slice_start
    hdr['slice_end'] = slice_end
    hdr['slice_duration'] = duration
    hdr['slice_code'] = slice_order_codes.code[label]