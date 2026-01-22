import importlib.util
import os
import re
import string
import subprocess
import sys
import sysconfig
import functools
from .errors import DistutilsPlatformError, DistutilsByteCompileError
from ._modified import newer
from .spawn import spawn
from ._log import log
from distutils.util import byte_compile
def get_macosx_target_ver():
    """Return the version of macOS for which we are building.

    The target version defaults to the version in sysconfig latched at time
    the Python interpreter was built, unless overridden by an environment
    variable. If neither source has a value, then None is returned"""
    syscfg_ver = get_macosx_target_ver_from_syscfg()
    env_ver = os.environ.get(MACOSX_VERSION_VAR)
    if env_ver:
        if syscfg_ver and split_version(syscfg_ver) >= [10, 3] and (split_version(env_ver) < [10, 3]):
            my_msg = '$' + MACOSX_VERSION_VAR + ' mismatch: now "%s" but "%s" during configure; must use 10.3 or later' % (env_ver, syscfg_ver)
            raise DistutilsPlatformError(my_msg)
        return env_ver
    return syscfg_ver