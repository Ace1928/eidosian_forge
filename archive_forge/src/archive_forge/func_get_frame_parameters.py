from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def get_frame_parameters(data):
    """
    Parse a zstd frame header into frame parameters.

    Depending on which fields are present in the frame and their values, the
    length of the frame parameters varies. If insufficient bytes are passed
    in to fully parse the frame parameters, ``ZstdError`` is raised. To ensure
    frame parameters can be parsed, pass in at least 18 bytes.

    :param data:
       Data from which to read frame parameters.
    :return:
       :py:class:`FrameParameters`
    """
    params = ffi.new('ZSTD_frameHeader *')
    data_buffer = ffi.from_buffer(data)
    zresult = lib.ZSTD_getFrameHeader(params, data_buffer, len(data_buffer))
    if lib.ZSTD_isError(zresult):
        raise ZstdError('cannot get frame parameters: %s' % _zstd_error(zresult))
    if zresult:
        raise ZstdError('not enough data for frame parameters; need %d bytes' % zresult)
    return FrameParameters(params[0])