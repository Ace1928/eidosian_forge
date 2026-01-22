import os
from hashlib import md5
import pytest
from fsspec.implementations.local import LocalFileSystem
from fsspec.tests.abstract.copy import AbstractCopyTests  # noqa
from fsspec.tests.abstract.get import AbstractGetTests  # noqa
from fsspec.tests.abstract.put import AbstractPutTests  # noqa
@pytest.fixture
def local_target(self, local_fs, local_join, local_path):
    """
        Return name of local directory that does not yet exist to copy into.

        Cleans up at the end of each test it which it is used.
        """
    target = local_join(local_path, 'target')
    yield target
    if local_fs.exists(target):
        local_fs.rm(target, recursive=True)