import pathlib
import platform
import sys
import pytest
from ..paths import file_uri_to_path, is_relative, normalized_uri
@pytest.mark.skipif(WIN, reason="can't test POSIX paths on Windows")
@pytest.mark.parametrize('root, path', [['~', '~/a'], ['~', '~/a/../b/'], ['/', '/'], ['/a', '/a/b'], ['/a', '/a/b/../c']])
def test_is_relative_ok(root, path):
    assert is_relative(root, path)