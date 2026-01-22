from __future__ import absolute_import
from .snappy import (
def get_decompress_function(specified_format, fin):
    if specified_format == FORMAT_AUTO:
        decompress_func, read_chunk = guess_format_by_header(fin)
        return (decompress_func, read_chunk)
    return (_DECOMPRESS_METHODS[specified_format], None)