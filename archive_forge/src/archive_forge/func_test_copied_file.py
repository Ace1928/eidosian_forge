import os
import stat
from dulwich import __version__ as dulwich_version
from dulwich.diff_tree import RenameDetector, tree_changes
from dulwich.index import IndexEntry, ConflictedIndexEntry
from dulwich.object_store import OverlayObjectStore
from dulwich.objects import S_IFGITLINK, ZERO_SHA, Blob, Tree
from ... import conflicts as _mod_conflicts
from ... import workingtree as _mod_workingtree
from ...bzr.inventorytree import InventoryTreeChange as TreeChange
from ...delta import TreeDelta
from ...tests import TestCase, TestCaseWithTransport
from ..mapping import default_mapping
from ..tree import tree_delta_from_git_changes
def test_copied_file(self):
    self.build_tree(['a'])
    self.wt.add(['a'])
    self.wt.copy_one('a', 'b')
    a = Blob.from_string(b'contents of a\n')
    self.store.add_object(a)
    oldt = Tree()
    oldt.add(b'a', stat.S_IFREG | 420, a.id)
    self.store.add_object(oldt)
    newt = Tree()
    newt.add(b'a', stat.S_IFREG | 420, a.id)
    newt.add(b'b', stat.S_IFREG | 420, a.id)
    self.store.add_object(newt)
    self.expectDelta([('modify', (b'', stat.S_IFDIR, oldt.id), (b'', stat.S_IFDIR, newt.id)), ('add', (None, None, None), (b'b', stat.S_IFREG | 420, a.id))], tree_id=oldt.id)
    if dulwich_version >= (0, 19, 15):
        self.expectDelta([('modify', (b'', stat.S_IFDIR, oldt.id), (b'', stat.S_IFDIR, newt.id)), ('copy', (b'a', stat.S_IFREG | 420, a.id), (b'b', stat.S_IFREG | 420, a.id))], tree_id=oldt.id, rename_detector=RenameDetector(self.store, find_copies_harder=True))
        self.expectDelta([('modify', (b'', stat.S_IFDIR, oldt.id), (b'', stat.S_IFDIR, newt.id)), ('add', (None, None, None), (b'b', stat.S_IFREG | 420, a.id))], tree_id=oldt.id, rename_detector=RenameDetector(self.store, find_copies_harder=False))