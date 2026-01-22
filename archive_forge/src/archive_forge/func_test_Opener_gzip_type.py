import contextlib
import hashlib
import os
import time
import unittest
from gzip import GzipFile
from io import BytesIO, UnsupportedOperation
from unittest import mock
import pytest
from packaging.version import Version
from ..deprecator import ExpiredDeprecationError
from ..openers import HAVE_INDEXED_GZIP, BZ2File, DeterministicGzipFile, ImageOpener, Opener
from ..optpkg import optional_package
from ..tmpdirs import InTemporaryDirectory
def test_Opener_gzip_type(tmp_path):
    data = b'this is some test data'
    fname = tmp_path / 'test.gz'
    with GzipFile(fname, mode='wb') as f:
        f.write(data)
    tests = [(False, {'mode': 'rb', 'keep_open': True}, GzipFile), (False, {'mode': 'rb', 'keep_open': False}, GzipFile), (False, {'mode': 'wb', 'keep_open': True}, GzipFile), (False, {'mode': 'wb', 'keep_open': False}, GzipFile), (True, {'mode': 'rb', 'keep_open': True}, MockIndexedGzipFile), (True, {'mode': 'rb', 'keep_open': False}, MockIndexedGzipFile), (True, {'mode': 'wb', 'keep_open': True}, GzipFile), (True, {'mode': 'wb', 'keep_open': False}, GzipFile)]
    for test in tests:
        igzip_present, kwargs, expected = test
        with patch_indexed_gzip(igzip_present):
            opener = Opener(fname, **kwargs)
            assert isinstance(opener.fobj, expected)
            del opener