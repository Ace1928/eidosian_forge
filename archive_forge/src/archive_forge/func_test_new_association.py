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
@mock.patch.dict('nibabel.openers.ImageOpener.compress_ext_map')
def test_new_association(self):

    def file_opener(fileish, mode):
        return open(fileish, mode)
    n_associations = len(ImageOpener.compress_ext_map)
    ImageOpener.compress_ext_map['.foo'] = (file_opener, ('mode',))
    assert n_associations + 1 == len(ImageOpener.compress_ext_map)
    assert '.foo' in ImageOpener.compress_ext_map
    with InTemporaryDirectory():
        with ImageOpener('test.foo', 'w'):
            pass
        assert os.path.exists('test.foo')
    assert '.foo' not in Opener.compress_ext_map