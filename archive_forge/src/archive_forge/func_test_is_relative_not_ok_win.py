import pathlib
import platform
import sys
import pytest
from ..paths import file_uri_to_path, is_relative, normalized_uri
@pytest.mark.skipif(not WIN, reason="can't test Windows paths on POSIX")
@pytest.mark.parametrize('root, path', [['c:\\Users\\user1', 'c:\\Users\\'], ['c:\\Users\\user1', 'd:\\'], ['c:\\Users', 'c:\\Users\\..']])
def test_is_relative_not_ok_win(root, path):
    assert not is_relative(root, path)