import gzip
import io
import struct
def zstd_encode(payload, level=None):
    if not has_zstd():
        raise NotImplementedError('Zstd codec is not available')
    if level is None:
        level = 3
    return bytes(cramjam.zstd.compress(payload, level=level))