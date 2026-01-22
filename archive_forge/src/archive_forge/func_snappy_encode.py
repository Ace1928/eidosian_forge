import gzip
import io
import struct
def snappy_encode(payload, xerial_compatible=True, xerial_blocksize=32 * 1024):
    """Encodes the given data with snappy compression.

    If xerial_compatible is set then the stream is encoded in a fashion
    compatible with the xerial snappy library.

    The block size (xerial_blocksize) controls how frequent the blocking occurs
    32k is the default in the xerial library.

    The format winds up being:


        +-------------+------------+--------------+------------+--------------+
        |   Header    | Block1 len | Block1 data  | Blockn len | Blockn data  |
        +-------------+------------+--------------+------------+--------------+
        |  16 bytes   |  BE int32  | snappy bytes |  BE int32  | snappy bytes |
        +-------------+------------+--------------+------------+--------------+


    It is important to note that the blocksize is the amount of uncompressed
    data presented to snappy at each block, whereas the blocklen is the number
    of bytes that will be present in the stream; so the length will always be
    <= blocksize.

    """
    if not has_snappy():
        raise NotImplementedError('Snappy codec is not available')
    if not xerial_compatible:
        return cramjam.snappy.compress_raw(payload)
    out = io.BytesIO()
    for fmt, dat in zip(_XERIAL_V1_FORMAT, _XERIAL_V1_HEADER):
        out.write(struct.pack('!' + fmt, dat))

    def chunker(payload, i, size):
        return memoryview(payload)[i:size + i]
    for chunk in (chunker(payload, i, xerial_blocksize) for i in range(0, len(payload), xerial_blocksize)):
        block = cramjam.snappy.compress_raw(chunk)
        block_size = len(block)
        out.write(struct.pack('!i', block_size))
        out.write(block)
    return out.getvalue()