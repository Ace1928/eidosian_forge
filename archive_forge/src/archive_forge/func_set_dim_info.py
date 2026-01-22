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
def set_dim_info(self, freq=None, phase=None, slice=None):
    """Sets nifti MRI slice etc dimension information

        Parameters
        ----------
        freq : {None, 0, 1, 2}
            axis of data array referring to frequency encoding
        phase : {None, 0, 1, 2}
            axis of data array referring to phase encoding
        slice : {None, 0, 1, 2}
            axis of data array referring to slice encoding

        ``None`` means the axis is not specified.

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_dim_info(1, 2, 0)
        >>> hdr.get_dim_info()
        (1, 2, 0)
        >>> hdr.set_dim_info(freq=1, phase=2, slice=0)
        >>> hdr.get_dim_info()
        (1, 2, 0)
        >>> hdr.set_dim_info()
        >>> hdr.get_dim_info()
        (None, None, None)
        >>> hdr.set_dim_info(freq=1, phase=None, slice=0)
        >>> hdr.get_dim_info()
        (1, None, 0)

        Notes
        -----
        This is stored in one byte in the header
        """
    for inp in (freq, phase, slice):
        if inp is not None and inp not in (0, 1, 2):
            raise HeaderDataError('Inputs must be in [None, 0, 1, 2]')
    info = 0
    if freq is not None:
        info = info | freq + 1 & 3
    if phase is not None:
        info = info | (phase + 1 & 3) << 2
    if slice is not None:
        info = info | (slice + 1 & 3) << 4
    self._structarr['dim_info'] = info