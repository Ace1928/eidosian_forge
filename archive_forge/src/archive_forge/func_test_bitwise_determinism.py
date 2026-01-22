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
def test_bitwise_determinism():
    with InTemporaryDirectory():
        msg = b"Hello, I'd like to have an argument."
        with open('ref.gz', 'wb') as fobj:
            with GzipFile(filename='', mode='wb', compresslevel=1, fileobj=fobj, mtime=0) as gzobj:
                gzobj.write(msg)
        anon_chksum = md5sum('ref.gz')
        now = time.time()
        with mock.patch('time.time') as t:
            t.return_value = now
            with Opener('a.gz', 'wb') as fobj:
                fobj.write(msg)
            t.return_value = now + 1
            with Opener('b.gz', 'wb') as fobj:
                fobj.write(msg)
        assert md5sum('a.gz') == anon_chksum
        assert md5sum('b.gz') == anon_chksum
        with Opener('filenameA.gz', 'wb', mtime=3405648064) as fobj:
            fobj.write(msg)
        with Opener('filenameB.gz', 'wb', mtime=3405648064) as fobj:
            fobj.write(msg)
        fnameA_chksum = md5sum('filenameA.gz')
        fnameB_chksum = md5sum('filenameB.gz')
        assert fnameA_chksum == fnameB_chksum != anon_chksum