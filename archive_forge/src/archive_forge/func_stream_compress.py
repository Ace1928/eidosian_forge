from __future__ import absolute_import
import struct
import cramjam
def stream_compress(src, dst, blocksize=_STREAM_TO_STREAM_BLOCK_SIZE, compressor_cls=StreamCompressor):
    """Takes an incoming file-like object and an outgoing file-like object,
    reads data from src, compresses it, and writes it to dst. 'src' should
    support the read method, and 'dst' should support the write method.

    The default blocksize is good for almost every scenario.
    """
    compressor = compressor_cls()
    while True:
        buf = src.read(blocksize)
        if not buf:
            break
        buf = compressor.add_chunk(buf)
        if buf:
            dst.write(buf)