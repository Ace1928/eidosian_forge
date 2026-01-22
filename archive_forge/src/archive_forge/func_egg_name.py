import sys
import os
import io
import time
import re
import types
from typing import Protocol
import zipfile
import zipimport
import warnings
import stat
import functools
import pkgutil
import operator
import platform
import collections
import plistlib
import email.parser
import errno
import tempfile
import textwrap
import inspect
import ntpath
import posixpath
import importlib
import importlib.machinery
from pkgutil import get_importer
import _imp
from os import utime
from os import open as os_open
from os.path import isdir, split
from pkg_resources.extern.jaraco.text import (
from pkg_resources.extern import platformdirs
from pkg_resources.extern import packaging
def egg_name(self):
    """Return what this distribution's standard .egg filename should be"""
    filename = '%s-%s-py%s' % (to_filename(self.project_name), to_filename(self.version), self.py_version or PY_MAJOR)
    if self.platform:
        filename += '-' + self.platform
    return filename