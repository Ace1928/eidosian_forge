import gzip
import io
import struct
def zstd_decode(payload):
    if not has_zstd():
        raise NotImplementedError('Zstd codec is not available')
    return bytes(cramjam.zstd.decompress(payload))