import os
from hashlib import md5
import pytest
from fsspec.implementations.local import LocalFileSystem
from fsspec.tests.abstract.copy import AbstractCopyTests  # noqa
from fsspec.tests.abstract.get import AbstractGetTests  # noqa
from fsspec.tests.abstract.put import AbstractPutTests  # noqa
@pytest.fixture
def local_10_files_with_hashed_names(self, local_fs, local_join, local_path):
    """
        Scenario on local filesystem that is used to check cp/get/put files order
        when source and destination are lists.

        Cleans up at the end of each test it which it is used.
        """
    source = self._10_files_with_hashed_names(local_fs, local_join, local_path)
    yield source
    local_fs.rm(source, recursive=True)