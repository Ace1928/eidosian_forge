import os
import time
from pathlib import Path
from unittest import mock, SkipTest
import pytest
from ...testing import TempFATFS
from ...utils.filemanip import (
def test_path_strict_resolve(tmpdir):
    """Check the monkeypatch to test strict resolution of Path."""
    tmpdir.chdir()
    testfile = Path('somefile.txt')
    resolved = '%s/somefile.txt' % tmpdir
    assert str(path_resolve(testfile)) == resolved
    assert str(path_resolve(testfile, strict=False)) == resolved
    with pytest.raises(FileNotFoundError):
        path_resolve(testfile, strict=True)
    open('somefile.txt', 'w').close()
    assert str(path_resolve(testfile, strict=True)) == resolved