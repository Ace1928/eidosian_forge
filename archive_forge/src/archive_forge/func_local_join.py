import os
from hashlib import md5
import pytest
from fsspec.implementations.local import LocalFileSystem
from fsspec.tests.abstract.copy import AbstractCopyTests  # noqa
from fsspec.tests.abstract.get import AbstractGetTests  # noqa
from fsspec.tests.abstract.put import AbstractPutTests  # noqa
@pytest.fixture
def local_join(self):
    """
        Return a function that joins its arguments together into a path, on
        the local filesystem.
        """
    return os.path.join