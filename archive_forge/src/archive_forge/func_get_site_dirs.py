from glob import glob
from distutils.util import get_platform
from distutils.util import convert_path, subst_vars
from distutils.errors import (
from distutils import log, dir_util
from distutils.command.build_scripts import first_line_re
from distutils.spawn import find_executable
from distutils.command import install
import sys
import os
from typing import Dict, List
import zipimport
import shutil
import tempfile
import zipfile
import re
import stat
import random
import textwrap
import warnings
import site
import struct
import contextlib
import subprocess
import shlex
import io
import configparser
import sysconfig
from sysconfig import get_path
from setuptools import Command
from setuptools.sandbox import run_setup
from setuptools.command import setopt
from setuptools.archive_util import unpack_archive
from setuptools.package_index import (
from setuptools.command import bdist_egg, egg_info
from setuptools.warnings import SetuptoolsDeprecationWarning, SetuptoolsWarning
from setuptools.wheel import Wheel
from pkg_resources import (
import pkg_resources
from ..compat import py39, py311
from .._path import ensure_directory
from ..extern.jaraco.text import yield_lines
def get_site_dirs():
    """
    Return a list of 'site' dirs
    """
    sitedirs = []
    sitedirs.extend(_pythonpath())
    prefixes = [sys.prefix]
    if sys.exec_prefix != sys.prefix:
        prefixes.append(sys.exec_prefix)
    for prefix in prefixes:
        if not prefix:
            continue
        if sys.platform in ('os2emx', 'riscos'):
            sitedirs.append(os.path.join(prefix, 'Lib', 'site-packages'))
        elif os.sep == '/':
            sitedirs.extend([os.path.join(prefix, 'lib', 'python{}.{}'.format(*sys.version_info), 'site-packages'), os.path.join(prefix, 'lib', 'site-python')])
        else:
            sitedirs.extend([prefix, os.path.join(prefix, 'lib', 'site-packages')])
        if sys.platform != 'darwin':
            continue
        if 'Python.framework' not in prefix:
            continue
        home = os.environ.get('HOME')
        if not home:
            continue
        home_sp = os.path.join(home, 'Library', 'Python', '{}.{}'.format(*sys.version_info), 'site-packages')
        sitedirs.append(home_sp)
    lib_paths = (get_path('purelib'), get_path('platlib'))
    sitedirs.extend((s for s in lib_paths if s not in sitedirs))
    if site.ENABLE_USER_SITE:
        sitedirs.append(site.USER_SITE)
    with contextlib.suppress(AttributeError):
        sitedirs.extend(site.getsitepackages())
    return list(map(normalize_path, sitedirs))