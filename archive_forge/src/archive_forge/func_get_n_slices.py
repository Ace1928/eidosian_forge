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
def get_n_slices(self):
    """Return the number of slices"""
    _, _, slice_dim = self.get_dim_info()
    if slice_dim is None:
        raise HeaderDataError('Slice dimension not set in header dim_info')
    shape = self.get_data_shape()
    try:
        slice_len = shape[slice_dim]
    except IndexError:
        raise HeaderDataError(f'Slice dimension index ({slice_dim}) outside shape tuple ({shape})')
    return slice_len