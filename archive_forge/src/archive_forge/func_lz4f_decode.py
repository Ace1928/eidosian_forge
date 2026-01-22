import gzip
import io
import struct
def lz4f_decode(payload):
    """Decode payload using interoperable LZ4 framing. Requires Kafka >= 0.10"""
    ctx = lz4f.createDecompContext()
    data = lz4f.decompressFrame(payload, ctx)
    lz4f.freeDecompContext(ctx)
    if data['next'] != 0:
        raise RuntimeError('lz4f unable to decompress full payload')
    return data['decomp']