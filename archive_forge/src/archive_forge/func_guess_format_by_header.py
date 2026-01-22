from __future__ import absolute_import
from .snappy import (
def guess_format_by_header(fin):
    """Tries to guess a compression format for the given input file by it's
    header.
    :return: tuple of decompression method and a chunk that was taken from the
        input for format detection.
    """
    chunk = None
    for check_method, decompress_func in _DECOMPRESS_FORMAT_FUNCS:
        ok, chunk = check_method(fin=fin, chunk=chunk)
        if not ok:
            continue
        return (decompress_func, chunk)
    raise UncompressError("Can't detect archive format")