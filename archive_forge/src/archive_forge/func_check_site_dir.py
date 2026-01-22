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
def check_site_dir(self):
    """Verify that self.install_dir is .pth-capable dir, if needed"""
    instdir = normalize_path(self.install_dir)
    pth_file = os.path.join(instdir, 'easy-install.pth')
    if not os.path.exists(instdir):
        try:
            os.makedirs(instdir)
        except OSError:
            self.cant_write_to_target()
    is_site_dir = instdir in self.all_site_dirs
    if not is_site_dir and (not self.multi_version):
        is_site_dir = self.check_pth_processing()
    else:
        testfile = self.pseudo_tempname() + '.write-test'
        test_exists = os.path.exists(testfile)
        try:
            if test_exists:
                os.unlink(testfile)
            open(testfile, 'wb').close()
            os.unlink(testfile)
        except OSError:
            self.cant_write_to_target()
    if not is_site_dir and (not self.multi_version):
        pythonpath = os.environ.get('PYTHONPATH', '')
        log.warn(self.__no_default_msg, self.install_dir, pythonpath)
    if is_site_dir:
        if self.pth_file is None:
            self.pth_file = PthDistributions(pth_file, self.all_site_dirs)
    else:
        self.pth_file = None
    if self.multi_version and (not os.path.exists(pth_file)):
        self.pth_file = None
    self.install_dir = instdir