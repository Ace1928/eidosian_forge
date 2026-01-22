from distutils.filelist import FileList as _FileList
from distutils.errors import DistutilsInternalError
from distutils.util import convert_path
from distutils import log
import distutils.errors
import distutils.filelist
import functools
import os
import re
import sys
import time
import collections
from .._importlib import metadata
from .. import _entry_points, _normalization
from . import _requirestxt
from setuptools import Command
from setuptools.command.sdist import sdist
from setuptools.command.sdist import walk_revctrl
from setuptools.command.setopt import edit_config
from setuptools.command import bdist_egg
import setuptools.unicode_utils as unicode_utils
from setuptools.glob import glob
from setuptools.extern import packaging
from ..warnings import SetuptoolsDeprecationWarning
def find_sources(self):
    """Generate SOURCES.txt manifest file"""
    manifest_filename = os.path.join(self.egg_info, 'SOURCES.txt')
    mm = manifest_maker(self.distribution)
    mm.ignore_egg_info_dir = self.ignore_egg_info_in_manifest
    mm.manifest = manifest_filename
    mm.run()
    self.filelist = mm.filelist