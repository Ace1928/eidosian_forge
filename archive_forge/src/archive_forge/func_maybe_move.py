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
def maybe_move(self, spec, dist_filename, setup_base):
    dst = os.path.join(self.build_directory, spec.key)
    if os.path.exists(dst):
        msg = '%r already exists in %s; build directory %s will not be kept'
        log.warn(msg, spec.key, self.build_directory, setup_base)
        return setup_base
    if os.path.isdir(dist_filename):
        setup_base = dist_filename
    else:
        if os.path.dirname(dist_filename) == setup_base:
            os.unlink(dist_filename)
        contents = os.listdir(setup_base)
        if len(contents) == 1:
            dist_filename = os.path.join(setup_base, contents[0])
            if os.path.isdir(dist_filename):
                setup_base = dist_filename
    ensure_directory(dst)
    shutil.move(setup_base, dst)
    return dst