import pyarrow as pa
from pyarrow import filesystem
import os
import pytest
def test_filesystem_deprecated():
    with pytest.warns(FutureWarning):
        filesystem.LocalFileSystem()
    with pytest.warns(FutureWarning):
        filesystem.LocalFileSystem.get_instance()