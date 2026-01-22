import email
import itertools
import functools
import os
import posixpath
import re
import zipfile
import contextlib
from distutils.util import get_platform
import setuptools
from setuptools.extern.packaging.version import Version as parse_version
from setuptools.extern.packaging.tags import sys_tags
from setuptools.extern.packaging.utils import canonicalize_name
from setuptools.command.egg_info import write_requirements, _egg_basename
from setuptools.archive_util import _unpack_zipfile_obj
def install_as_egg(self, destination_eggdir):
    """Install wheel as an egg directory."""
    with zipfile.ZipFile(self.filename) as zf:
        self._install_as_egg(destination_eggdir, zf)