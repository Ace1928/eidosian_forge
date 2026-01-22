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
def executable_filename(self, basename, strip_dir=0, output_dir=''):
    assert output_dir is not None
    if strip_dir:
        basename = os.path.basename(basename)
    return os.path.join(output_dir, basename + (self.exe_extension or ''))