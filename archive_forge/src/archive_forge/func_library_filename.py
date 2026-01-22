import sys
import os
import re
import warnings
from .errors import (
from .spawn import spawn
from .file_util import move_file
from .dir_util import mkpath
from ._modified import newer_group
from .util import split_quoted, execute
from ._log import log
def library_filename(self, libname, lib_type='static', strip_dir=0, output_dir=''):
    assert output_dir is not None
    expected = '"static", "shared", "dylib", "xcode_stub"'
    if lib_type not in eval(expected):
        raise ValueError(f"'lib_type' must be {expected}")
    fmt = getattr(self, lib_type + '_lib_format')
    ext = getattr(self, lib_type + '_lib_extension')
    dir, base = os.path.split(libname)
    filename = fmt % (base, ext)
    if strip_dir:
        dir = ''
    return os.path.join(output_dir, dir, filename)