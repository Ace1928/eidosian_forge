from ctypes import POINTER, c_bool, c_char_p, c_double, c_int, c_int64, c_void_p
from functools import partial
from django.contrib.gis.gdal.prototypes.errcheck import (
def string_output(func, argtypes, offset=-1, str_result=False, decoding=None):
    """
    Generate a ctypes prototype for the given function with the
    given argument types that returns a string from a GDAL pointer.
    The `const` flag indicates whether the allocated pointer should
    be freed via the GDAL library routine VSIFree -- but only applies
    only when `str_result` is True.
    """
    func.argtypes = argtypes
    if str_result:
        func.restype = gdal_char_p
    else:
        func.restype = c_int

    def _check_str(result, func, cargs):
        res = check_string(result, func, cargs, offset=offset, str_result=str_result)
        if res and decoding:
            res = res.decode(decoding)
        return res
    func.errcheck = _check_str
    return func