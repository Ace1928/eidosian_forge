from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_constants import DebugInfoHolder, IS_WINDOWS, IS_JYTHON, \
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydevd_bundle.pydevd_comm_constants import file_system_encoding, filesystem_encoding_is_utf8
from _pydev_bundle.pydev_log import error_once
import json
import os.path
import sys
import itertools
import ntpath
from functools import partial
def get_abs_path_real_path_and_base_from_frame(frame, NORM_PATHS_AND_BASE_CONTAINER=NORM_PATHS_AND_BASE_CONTAINER):
    try:
        return NORM_PATHS_AND_BASE_CONTAINER[frame.f_code.co_filename]
    except:
        f = frame.f_code.co_filename
        if f is not None and f.startswith(('build/bdist.', 'build\\bdist.')):
            f = frame.f_globals['__file__']
        if get_abs_path_real_path_and_base_from_file is None:
            if not f:
                f = '<string>'
            i = max(f.rfind('/'), f.rfind('\\'))
            return (f, f, f[i + 1:])
        ret = get_abs_path_real_path_and_base_from_file(f)
        NORM_PATHS_AND_BASE_CONTAINER[frame.f_code.co_filename] = ret
        return ret