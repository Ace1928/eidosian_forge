from __future__ import unicode_literals
import os
import posixpath
from contextlib import contextmanager
from shutil import rmtree
from tempfile import mkdtemp
import pytest
from pybtex import errors, io
from .utils import diff, get_data
def group_by_suffix(filenames):
    filenames_by_suffix = dict(((posixpath.splitext(filename)[1], filename) for filename in filenames))
    return filenames_by_suffix