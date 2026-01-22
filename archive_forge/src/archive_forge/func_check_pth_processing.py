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
def check_pth_processing(self):
    """Empirically verify whether .pth files are supported in inst. dir"""
    instdir = self.install_dir
    log.info('Checking .pth file support in %s', instdir)
    pth_file = self.pseudo_tempname() + '.pth'
    ok_file = pth_file + '.ok'
    ok_exists = os.path.exists(ok_file)
    tmpl = _one_liner('\n            import os\n            f = open({ok_file!r}, \'w\', encoding="utf-8")\n            f.write(\'OK\')\n            f.close()\n            ') + '\n'
    try:
        if ok_exists:
            os.unlink(ok_file)
        dirname = os.path.dirname(ok_file)
        os.makedirs(dirname, exist_ok=True)
        f = open(pth_file, 'w', encoding=py39.LOCALE_ENCODING)
    except OSError:
        self.cant_write_to_target()
    else:
        try:
            f.write(tmpl.format(**locals()))
            f.close()
            f = None
            executable = sys.executable
            if os.name == 'nt':
                dirname, basename = os.path.split(executable)
                alt = os.path.join(dirname, 'pythonw.exe')
                use_alt = basename.lower() == 'python.exe' and os.path.exists(alt)
                if use_alt:
                    executable = alt
            from distutils.spawn import spawn
            spawn([executable, '-E', '-c', 'pass'], 0)
            if os.path.exists(ok_file):
                log.info('TEST PASSED: %s appears to support .pth files', instdir)
                return True
        finally:
            if f:
                f.close()
            if os.path.exists(ok_file):
                os.unlink(ok_file)
            if os.path.exists(pth_file):
                os.unlink(pth_file)
    if not self.multi_version:
        log.warn('TEST FAILED: %s does NOT support .pth files', instdir)
    return False