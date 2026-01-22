from gitdb.test.lib import (
from gitdb.stream import DeltaApplyReader
from gitdb.pack import (
from gitdb.base import (
from gitdb.fun import delta_types
from gitdb.exc import UnsupportedOperation
from gitdb.util import to_bin_sha
import pytest
import os
import tempfile
def test_pack_index(self):
    for indexfile, version, size in (self.packindexfile_v1, self.packindexfile_v2):
        index = PackIndexFile(indexfile)
        self._assert_index_file(index, version, size)