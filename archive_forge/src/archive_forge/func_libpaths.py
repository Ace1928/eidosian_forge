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
def libpaths(paths, bits):
    """Return a list of library paths valid on 32 or 64 bit systems.

    Inputs:
      paths : sequence
        A sequence of strings (typically paths)
      bits : int
        An integer, the only valid values are 32 or 64.  A ValueError exception
      is raised otherwise.

    Examples:

    Consider a list of directories
    >>> paths = ['/usr/X11R6/lib','/usr/X11/lib','/usr/lib']

    For a 32-bit platform, this is already valid:
    >>> np.distutils.system_info.libpaths(paths,32)
    ['/usr/X11R6/lib', '/usr/X11/lib', '/usr/lib']

    On 64 bits, we prepend the '64' postfix
    >>> np.distutils.system_info.libpaths(paths,64)
    ['/usr/X11R6/lib64', '/usr/X11R6/lib', '/usr/X11/lib64', '/usr/X11/lib',
    '/usr/lib64', '/usr/lib']
    """
    if bits not in (32, 64):
        raise ValueError('Invalid bit size in libpaths: 32 or 64 only')
    if bits == 32:
        return paths
    out = []
    for p in paths:
        out.extend([p + '64', p])
    return out