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
def show_all(argv=None):
    import inspect
    if argv is None:
        argv = sys.argv
    opts, args = parseCmdLine(argv)
    if opts.verbose:
        log.set_threshold(log.DEBUG)
    else:
        log.set_threshold(log.INFO)
    show_only = []
    for n in args:
        if n[-5:] != '_info':
            n = n + '_info'
        show_only.append(n)
    show_all = not show_only
    _gdict_ = globals().copy()
    for name, c in _gdict_.items():
        if not inspect.isclass(c):
            continue
        if not issubclass(c, system_info) or c is system_info:
            continue
        if not show_all:
            if name not in show_only:
                continue
            del show_only[show_only.index(name)]
        conf = c()
        conf.verbosity = 2
        conf.get_info()
    if show_only:
        log.info('Info classes not defined: %s', ','.join(show_only))