import gzip
import io
import struct
def has_lz4():
    if lz4 is not None:
        return True
    if lz4f is not None:
        return True
    if lz4framed is not None:
        return True
    return False