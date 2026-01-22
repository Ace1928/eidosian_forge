import os
import pytest
from monty.os import cd, makedirs_p
from monty.os.path import find_exts, zpath
def test_find_exts(self):
    assert len(find_exts(os.path.dirname(__file__), 'py')) >= 18
    assert len(find_exts(os.path.dirname(__file__), 'bz2')) == 2
    assert len(find_exts(os.path.dirname(__file__), 'bz2', exclude_dirs='test_files')) == 0
    assert len(find_exts(os.path.dirname(__file__), 'bz2', include_dirs='test_files')) == 2