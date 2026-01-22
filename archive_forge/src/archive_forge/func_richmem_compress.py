from .zstdfile import *
from .seekable_zstdfile import *
def richmem_compress(data, level_or_option=None, zstd_dict=None):
    """Compress a block of data, return a bytes object.

    Use rich memory mode, it's faster than compress() in some cases, but
    allocates more memory.

    Compressing b'' will get an empty content frame (9 bytes or more).

    Parameters
    data:            A bytes-like object, data to be compressed.
    level_or_option: When it's an int object, it represents compression level.
                     When it's a dict object, it contains advanced compression
                     parameters.
    zstd_dict:       A ZstdDict object, pre-trained dictionary for compression.
    """
    comp = RichMemZstdCompressor(level_or_option, zstd_dict)
    return comp.compress(data)