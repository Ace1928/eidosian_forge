import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
@pytest.mark.parametrize('filename, split', [('foo.nii', ('', 'foo', '.nii')), ('foo.nii.gz', ('', 'foo', '.nii.gz')), ('foo.niml.dset', ('', 'foo', '.niml.dset')), ('/usr/local/foo.nii.gz', ('/usr/local', 'foo', '.nii.gz')), ('../usr/local/foo.nii', ('../usr/local', 'foo', '.nii')), ('/usr/local/foo.a.b.c.d', ('/usr/local', 'foo.a.b.c', '.d')), ('/usr/local/', ('/usr/local', '', ''))])
def test_split_filename(filename, split):
    res = split_filename(filename)
    assert res == split