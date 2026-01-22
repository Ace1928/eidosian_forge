from distutils.dir_util import remove_tree, mkpath
from distutils import log
from types import CodeType
import sys
import os
import re
import textwrap
import marshal
from setuptools.extension import Library
from setuptools import Command
from .._path import ensure_directory
from sysconfig import get_path, get_python_version
def call_command(self, cmdname, **kw):
    """Invoke reinitialized command `cmdname` with keyword args"""
    for dirname in INSTALL_DIRECTORY_ATTRS:
        kw.setdefault(dirname, self.bdist_dir)
    kw.setdefault('skip_build', self.skip_build)
    kw.setdefault('dry_run', self.dry_run)
    cmd = self.reinitialize_command(cmdname, **kw)
    self.run_command(cmdname)
    return cmd