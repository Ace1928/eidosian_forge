import os
import shutil
import stat
from dulwich.objects import Blob, Tree
from ...branchbuilder import BranchBuilder
from ...bzr.inventory import InventoryDirectory, InventoryFile
from ...errors import NoSuchRevision
from ...graph import DictParentsProvider, Graph
from ...tests import TestCase, TestCaseWithTransport
from ...tests.features import SymlinkFeature
from ..cache import DictGitShaMap
from ..object_store import (BazaarObjectStore, LRUTreeCache,
def test_get_blob(self):
    self.branch.lock_write()
    self.addCleanup(self.branch.unlock)
    b = Blob()
    b.data = b'a\nb\nc\nd\ne\n'
    self.store.lock_read()
    self.addCleanup(self.store.unlock)
    self.assertRaises(KeyError, self.store.__getitem__, b.id)
    bb = BranchBuilder(branch=self.branch)
    bb.start_series()
    bb.build_snapshot(None, [('add', ('', None, 'directory', None)), ('add', ('foo', b'foo-id', 'file', b'a\nb\nc\nd\ne\n'))])
    bb.finish_series()
    self.assertRaises(KeyError, self.store.__getitem__, b.id)
    self.store.unlock()
    self.store.lock_read()
    self.assertEqual(b, self.store[b.id])