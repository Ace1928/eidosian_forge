from __future__ import annotations
import numpy as np
from .arrayproxy import ArrayProxy
from .arraywriters import ArrayWriter, WriterError, get_slope_inter, make_array_writer
from .batteryrunners import Report
from .fileholders import copy_file_map
from .spatialimages import HeaderDataError, HeaderTypeError, SpatialHeader, SpatialImage
from .volumeutils import (
from .wrapstruct import LabeledWrapStruct
def data_to_fileobj(self, data, fileobj, rescale=True):
    """Write `data` to `fileobj`, maybe rescaling data, modifying `self`

        In writing the data, we match the header to the written data, by
        setting the header scaling factors, iff `rescale` is True.  Thus we
        modify `self` in the process of writing the data.

        Parameters
        ----------
        data : array-like
           data to write; should match header defined shape
        fileobj : file-like object
           Object with file interface, implementing ``write`` and
           ``seek``
        rescale : {True, False}, optional
            Whether to try and rescale data to match output dtype specified by
            header. If True and scaling needed and header cannot scale, then
            raise ``HeaderTypeError``.

        Examples
        --------
        >>> from nibabel.analyze import AnalyzeHeader
        >>> hdr = AnalyzeHeader()
        >>> hdr.set_data_shape((1, 2, 3))
        >>> hdr.set_data_dtype(np.float64)
        >>> from io import BytesIO
        >>> str_io = BytesIO()
        >>> data = np.arange(6).reshape(1,2,3)
        >>> hdr.data_to_fileobj(data, str_io)
        >>> data.astype(np.float64).tobytes('F') == str_io.getvalue()
        True
        """
    data = np.asanyarray(data)
    shape = self.get_data_shape()
    if data.shape != shape:
        raise HeaderDataError('Data should be shape (%s)' % ', '.join((str(s) for s in shape)))
    out_dtype = self.get_data_dtype()
    if rescale:
        try:
            arr_writer = make_array_writer(data, out_dtype, self.has_data_slope, self.has_data_intercept)
        except WriterError as e:
            raise HeaderTypeError(str(e))
    else:
        arr_writer = ArrayWriter(data, out_dtype, check_scaling=False)
    seek_tell(fileobj, self.get_data_offset())
    arr_writer.to_fileobj(fileobj)
    self.set_slope_inter(*get_slope_inter(arr_writer))