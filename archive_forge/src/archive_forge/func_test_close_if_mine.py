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
def test_close_if_mine():
    with InTemporaryDirectory():
        sobj = BytesIO()
        lunk = Lunk('')
        for input in ('test.txt', 'test.txt.gz', 'test.txt.bz2', sobj, lunk):
            fobj = Opener(input, 'wb')
            has_closed = hasattr(fobj.fobj, 'closed')
            if has_closed:
                assert not fobj.closed
            fobj.close_if_mine()
            is_str = type(input) is str
            if has_closed:
                assert fobj.closed == is_str