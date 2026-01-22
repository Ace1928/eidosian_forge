from ctypes import (
import ctypes
from ctypes.util import find_library
import logging
import mmap
import os
import sysconfig
from .exception import ArchiveError
def get_write_filter_function(filter_name):
    function_name = 'write_add_filter_' + filter_name
    func = globals().get(function_name)
    if func:
        return func
    try:
        return ffi(function_name, [c_archive_p], c_int, check_int)
    except AttributeError:
        raise ValueError('the write filter %r is not available' % filter_name)