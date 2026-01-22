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
def test_against_empty_tree(self):
    i = self.get_simple_index('index')
    changes = list(i.changes_from_tree(MemoryObjectStore(), None))
    self.assertEqual(1, len(changes))
    (oldname, newname), (oldmode, newmode), (oldsha, newsha) = changes[0]
    self.assertEqual(b'bla', newname)
    self.assertEqual(b'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391', newsha)