import sys
import sysconfig
import os
import re
from distutils import log
from distutils.core import Command
from distutils.debug import DEBUG
from distutils.sysconfig import get_config_vars
from distutils.errors import DistutilsPlatformError
from distutils.file_util import write_file
from distutils.util import convert_path, subst_vars, change_root
from distutils.util import get_platform
from distutils.errors import DistutilsOptionError
from site import USER_BASE
from site import USER_SITE
def handle_extra_path(self):
    """Set `path_file` and `extra_dirs` using `extra_path`."""
    if self.extra_path is None:
        self.extra_path = self.distribution.extra_path
    if self.extra_path is not None:
        log.warn('Distribution option extra_path is deprecated. See issue27919 for details.')
        if isinstance(self.extra_path, str):
            self.extra_path = self.extra_path.split(',')
        if len(self.extra_path) == 1:
            path_file = extra_dirs = self.extra_path[0]
        elif len(self.extra_path) == 2:
            path_file, extra_dirs = self.extra_path
        else:
            raise DistutilsOptionError("'extra_path' option must be a list, tuple, or comma-separated string with 1 or 2 elements")
        extra_dirs = convert_path(extra_dirs)
    else:
        path_file = None
        extra_dirs = ''
    self.path_file = path_file
    self.extra_dirs = extra_dirs