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
def get_entry_map(self, group=None):
    """Return the entry point map for `group`, or the full entry map"""
    try:
        ep_map = self._ep_map
    except AttributeError:
        ep_map = self._ep_map = EntryPoint.parse_map(self._get_metadata('entry_points.txt'), self)
    if group is not None:
        return ep_map.get(group, {})
    return ep_map