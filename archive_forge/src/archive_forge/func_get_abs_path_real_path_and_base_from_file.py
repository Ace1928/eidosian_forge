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
def get_abs_path_real_path_and_base_from_file(filename, NORM_PATHS_AND_BASE_CONTAINER=NORM_PATHS_AND_BASE_CONTAINER):
    try:
        return NORM_PATHS_AND_BASE_CONTAINER[filename]
    except:
        f = filename
        if not f:
            f = '<string>'
        if f.startswith('<'):
            return (f, normcase(f), f)
        if _abs_and_canonical_path is None:
            i = max(f.rfind('/'), f.rfind('\\'))
            return (f, f, f[i + 1:])
        if f is not None:
            if f.endswith('.pyc'):
                f = f[:-1]
            elif f.endswith('$py.class'):
                f = f[:-len('$py.class')] + '.py'
        abs_path, canonical_normalized_filename = _abs_and_canonical_path(f)
        try:
            base = os_path_basename(canonical_normalized_filename)
        except AttributeError:
            i = max(f.rfind('/'), f.rfind('\\'))
            base = f[i + 1:]
        ret = (abs_path, canonical_normalized_filename, base)
        NORM_PATHS_AND_BASE_CONTAINER[filename] = ret
        return ret