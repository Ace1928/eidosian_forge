import ast
import os
import re
from hacking import core
from os_win.utils.winapi import libs as w_lib
import_translation_for_log_or_exception = re.compile(
@core.flake8ext
def assert_ctypes_libs_not_used_directly(logical_line, filename):
    w_lib_path = os.path.join(*w_lib.__name__.split('.'))
    if w_lib_path in filename:
        return
    res = ctypes_external_lib_re.search(logical_line)
    if res:
        yield (0, 'O301: Using external libraries via ctypes directly is not allowed. Please use the following function to retrieve a supported library handle: %s.get_shared_lib_handle' % w_lib.__name__)