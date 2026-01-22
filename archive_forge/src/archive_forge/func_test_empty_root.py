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
def test_empty_root(self):
    child_ie = InventoryDirectory(b'bar', 'bar', b'bar')
    t = directory_to_tree('', [child_ie], lambda p, x: None, {}, None, allow_empty=True)
    self.assertEqual(Tree(), t)