import os
from hashlib import md5
import pytest
from fsspec.implementations.local import LocalFileSystem
from fsspec.tests.abstract.copy import AbstractCopyTests  # noqa
from fsspec.tests.abstract.get import AbstractGetTests  # noqa
from fsspec.tests.abstract.put import AbstractPutTests  # noqa
@pytest.fixture
def fs_target(self, fs, fs_join, fs_path):
    """
        Return name of remote directory that does not yet exist to copy into.

        Cleans up at the end of each test it which it is used.
        """
    target = fs_join(fs_path, 'target')
    yield target
    if fs.exists(target):
        fs.rm(target, recursive=True)