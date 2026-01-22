from __future__ import unicode_literals
import os
import posixpath
from contextlib import contextmanager
from shutil import rmtree
from tempfile import mkdtemp
import pytest
from pybtex import errors, io
from .utils import diff, get_data
@contextmanager
def cd_tempdir():
    current_workdir = os.getcwd()
    tempdir = mkdtemp(prefix='pybtex_test_')
    os.chdir(tempdir)
    try:
        yield tempdir
    finally:
        os.chdir(current_workdir)
        rmtree(tempdir)