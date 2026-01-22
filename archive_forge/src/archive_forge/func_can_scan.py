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
def can_scan():
    if not sys.platform.startswith('java') and sys.platform != 'cli':
        return True
    log.warn('Unable to analyze compiled code on this platform.')
    log.warn("Please ask the author to include a 'zip_safe' setting (either True or False) in the package's setup.py")