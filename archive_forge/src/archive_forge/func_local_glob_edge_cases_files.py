import os
from hashlib import md5
import pytest
from fsspec.implementations.local import LocalFileSystem
from fsspec.tests.abstract.copy import AbstractCopyTests  # noqa
from fsspec.tests.abstract.get import AbstractGetTests  # noqa
from fsspec.tests.abstract.put import AbstractPutTests  # noqa
@pytest.fixture
def local_glob_edge_cases_files(self, local_fs, local_join, local_path):
    """
        Scenario on local filesystem that is used for glob edge cases cp/get/put tests.

        Cleans up at the end of each test it which it is used.
        """
    source = self._glob_edge_cases_files(local_fs, local_join, local_path)
    yield source
    local_fs.rm(source, recursive=True)