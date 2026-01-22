import sys
import os
import re
import copy
import warnings
import subprocess
import textwrap
from glob import glob
from functools import reduce
from configparser import NoOptionError
from configparser import RawConfigParser as ConfigParser
from distutils.errors import DistutilsError
from distutils.dist import Distribution
import sysconfig
from numpy.distutils import log
from distutils.util import get_platform
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import (is_sequence, is_string,
from numpy.distutils.command.config import config as cmd_config
from numpy.distutils import customized_ccompiler as _customized_ccompiler
from numpy.distutils import _shell_utils
import distutils.ccompiler
import tempfile
import shutil
import platform
def check_libs(self, lib_dirs, libs, opt_libs=[]):
    """If static or shared libraries are available then return
        their info dictionary.

        Checks for all libraries as shared libraries first, then
        static (or vice versa if self.search_static_first is True).
        """
    exts = self.library_extensions()
    info = None
    for ext in exts:
        info = self._check_libs(lib_dirs, libs, opt_libs, [ext])
        if info is not None:
            break
    if not info:
        log.info('  libraries %s not found in %s', ','.join(libs), lib_dirs)
    return info