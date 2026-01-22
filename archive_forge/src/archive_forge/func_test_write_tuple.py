import os
import shutil
import stat
import struct
import sys
import tempfile
from io import BytesIO
from dulwich.tests import TestCase, skipIf
from ..index import (
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..repo import Repo
def test_write_tuple(self):
    f = BytesIO()
    write_cache_time(f, (434343, 21))
    self.assertEqual(struct.pack('>LL', 434343, 21), f.getvalue())