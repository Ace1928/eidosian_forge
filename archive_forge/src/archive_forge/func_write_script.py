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
def write_script(self, script_name, contents, mode='t', blockers=()):
    """Write an executable file to the scripts directory"""
    self.delete_blockers([os.path.join(self.script_dir, x) for x in blockers])
    log.info('Installing %s script to %s', script_name, self.script_dir)
    target = os.path.join(self.script_dir, script_name)
    self.add_output(target)
    if self.dry_run:
        return
    mask = current_umask()
    ensure_directory(target)
    if os.path.exists(target):
        os.unlink(target)
    with open(target, 'w' + mode) as f:
        f.write(contents)
    chmod(target, 511 - mask)