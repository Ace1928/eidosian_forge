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
def get_host_platform():
    """
    Return a string that identifies the current platform. Use this
    function to distinguish platform-specific build directories and
    platform-specific built distributions.
    """
    if sys.version_info < (3, 8):
        if os.name == 'nt':
            if '(arm)' in sys.version.lower():
                return 'win-arm32'
            if '(arm64)' in sys.version.lower():
                return 'win-arm64'
    if sys.version_info < (3, 9):
        if os.name == 'posix' and hasattr(os, 'uname'):
            osname, host, release, version, machine = os.uname()
            if osname[:3] == 'aix':
                from .py38compat import aix_platform
                return aix_platform(osname, version, release)
    return sysconfig.get_platform()