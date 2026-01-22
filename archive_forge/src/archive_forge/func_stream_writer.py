from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def stream_writer(self, writer, write_size=DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE, write_return_read=True, closefd=True):
    """
        Push-based stream wrapper that performs decompression.

        This method constructs a stream wrapper that conforms to the
        ``io.RawIOBase`` interface and performs transparent decompression
        when writing to a wrapper stream.

        See :py:class:`zstandard.ZstdDecompressionWriter` for more documentation
        and usage examples.

        :param writer:
           Destination for decompressed output. Can be any object with a
           ``write(data)``.
        :param write_size:
           Integer size of chunks to ``write()`` to ``writer``.
        :param write_return_read:
           Whether ``write()`` should return the number of bytes of input
           consumed. If False, ``write()`` returns the number of bytes sent
           to the inner stream.
        :param closefd:
           Whether to ``close()`` the inner stream when this stream is closed.
        :return:
           :py:class:`zstandard.ZstdDecompressionWriter`
        """
    if not hasattr(writer, 'write'):
        raise ValueError('must pass an object with a write() method')
    return ZstdDecompressionWriter(self, writer, write_size, write_return_read, closefd=closefd)