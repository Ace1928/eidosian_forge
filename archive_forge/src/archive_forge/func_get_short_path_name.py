from __future__ import unicode_literals
import os.path as op
from send2trash.compat import text_type
from send2trash.util import preprocess_paths
from ctypes import (
from ctypes.wintypes import HWND, UINT, LPCWSTR, BOOL
def get_short_path_name(long_name):
    prefix, long_path = prefix_and_path(long_name)
    buf_size = GetShortPathNameW(long_path, None, 0)
    if not buf_size:
        err_no = GetLastError()
        raise WindowsError(err_no, FormatError(err_no), long_path)
    output = create_unicode_buffer(buf_size)
    GetShortPathNameW(long_path, output, buf_size)
    return get_awaited_path_from_prefix(prefix, output.value)