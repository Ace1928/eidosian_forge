from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def compressobj(self, size=-1):
    """
        Obtain a compressor exposing the Python standard library compression API.

        See :py:class:`ZstdCompressionObj` for the full documentation.

        :param size:
           Size in bytes of data that will be compressed.
        :return:
           :py:class:`ZstdCompressionObj`
        """
    lib.ZSTD_CCtx_reset(self._cctx, lib.ZSTD_reset_session_only)
    if size < 0:
        size = lib.ZSTD_CONTENTSIZE_UNKNOWN
    zresult = lib.ZSTD_CCtx_setPledgedSrcSize(self._cctx, size)
    if lib.ZSTD_isError(zresult):
        raise ZstdError('error setting source size: %s' % _zstd_error(zresult))
    cobj = ZstdCompressionObj()
    cobj._out = ffi.new('ZSTD_outBuffer *')
    cobj._dst_buffer = ffi.new('char[]', COMPRESSION_RECOMMENDED_OUTPUT_SIZE)
    cobj._out.dst = cobj._dst_buffer
    cobj._out.size = COMPRESSION_RECOMMENDED_OUTPUT_SIZE
    cobj._out.pos = 0
    cobj._compressor = self
    cobj._finished = False
    return cobj