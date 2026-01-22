import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
@pytest.mark.parametrize('file, length, expected_files', [('/path/test.img', 3, ['/path/test.hdr', '/path/test.img', '/path/test.mat']), ('/path/test.hdr', 3, ['/path/test.hdr', '/path/test.img', '/path/test.mat']), ('/path/test.BRIK', 2, ['/path/test.BRIK', '/path/test.HEAD']), ('/path/test.HEAD', 2, ['/path/test.BRIK', '/path/test.HEAD']), ('/path/foo.nii', 2, ['/path/foo.nii', '/path/foo.mat'])])
def test_related_files(file, length, expected_files):
    related_files = get_related_files(file)
    assert len(related_files) == length
    for ef in expected_files:
        assert ef in related_files