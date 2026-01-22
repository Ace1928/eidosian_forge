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
def get_resource_filename(self, manager, resource_name):
    if not self.egg_name:
        raise NotImplementedError('resource_filename() only supported for .egg, not .zip')
    zip_path = self._resource_to_zip(resource_name)
    eagers = self._get_eager_resources()
    if '/'.join(self._parts(zip_path)) in eagers:
        for name in eagers:
            self._extract_resource(manager, self._eager_to_zip(name))
    return self._extract_resource(manager, zip_path)