import struct
import zlib
def chunks_to_gzip(chunks, factory=zlib.compressobj, level=zlib.Z_DEFAULT_COMPRESSION, method=zlib.DEFLATED, width=-zlib.MAX_WBITS, mem=zlib.DEF_MEM_LEVEL, crc32=zlib.crc32):
    """Create a gzip file containing chunks and return its content.

    :param chunks: An iterable of strings. Each string can have arbitrary
        layout.
    """
    result = [b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff']
    compress = factory(level, method, width, mem, 0)
    crc = 0
    total_len = 0
    for chunk in chunks:
        crc = crc32(chunk, crc)
        total_len += len(chunk)
        zbytes = compress.compress(chunk)
        if zbytes:
            result.append(zbytes)
    result.append(compress.flush())
    result.append(struct.pack('<LL', LOWU32(crc), LOWU32(total_len)))
    return result