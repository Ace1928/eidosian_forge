from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def chunker(self, size=-1, chunk_size=COMPRESSION_RECOMMENDED_OUTPUT_SIZE):
    """
        Create an object for iterative compressing to same-sized chunks.

        This API is similar to :py:meth:`ZstdCompressor.compressobj` but has
        better performance properties.

        :param size:
           Size in bytes of data that will be compressed.
        :param chunk_size:
           Size of compressed chunks.
        :return:
           :py:class:`ZstdCompressionChunker`
        """
    lib.ZSTD_CCtx_reset(self._cctx, lib.ZSTD_reset_session_only)
    if size < 0:
        size = lib.ZSTD_CONTENTSIZE_UNKNOWN
    zresult = lib.ZSTD_CCtx_setPledgedSrcSize(self._cctx, size)
    if lib.ZSTD_isError(zresult):
        raise ZstdError('error setting source size: %s' % _zstd_error(zresult))
    return ZstdCompressionChunker(self, chunk_size=chunk_size)