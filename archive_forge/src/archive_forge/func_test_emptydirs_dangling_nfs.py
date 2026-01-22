import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
def test_emptydirs_dangling_nfs(tmp_path):
    busyfile = tmp_path / 'base' / 'subdir' / 'busyfile'
    busyfile.parent.mkdir(parents=True)
    busyfile.touch()
    with mock.patch('os.unlink') as mocked:
        mocked.side_effect = nfs_unlink
        emptydirs(tmp_path / 'base')
    assert Path.exists(tmp_path / 'base')
    assert not busyfile.exists()
    assert busyfile.parent.exists()