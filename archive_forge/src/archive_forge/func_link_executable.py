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
def link_executable(self, objects, output_progname, output_dir=None, libraries=None, library_dirs=None, runtime_library_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, target_lang=None):
    self.link(CCompiler.EXECUTABLE, objects, self.executable_filename(output_progname), output_dir, libraries, library_dirs, runtime_library_dirs, None, debug, extra_preargs, extra_postargs, None, target_lang)