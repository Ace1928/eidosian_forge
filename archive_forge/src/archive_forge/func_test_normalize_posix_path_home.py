import pathlib
import platform
import sys
import pytest
from ..paths import file_uri_to_path, is_relative, normalized_uri
@pytest.mark.skipif(WIN, reason="can't test POSIX paths on Windows")
@pytest.mark.parametrize('root_dir, expected_root_uri', [['~', HOME.as_uri()]])
def test_normalize_posix_path_home(root_dir, expected_root_uri):
    assert normalized_uri(root_dir) == expected_root_uri