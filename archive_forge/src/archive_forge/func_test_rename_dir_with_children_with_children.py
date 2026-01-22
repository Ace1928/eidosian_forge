import os
from breezy import tests
from breezy.bzr import inventory
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_dir_with_children_with_children(self):
    wt = self.make_branch_and_tree('.')
    wt.lock_write()
    root_id = wt.path2id('')
    self.addCleanup(wt.unlock)
    self.build_tree(['foo/', 'foo/bar/', 'foo/bar/baz'])
    wt.add(['foo', 'foo/bar', 'foo/bar/baz'], ids=[b'foo-id', b'bar-id', b'baz-id'])
    wt.apply_inventory_delta([('foo', 'quux', b'foo-id', inventory.InventoryDirectory(b'foo-id', 'quux', root_id))])
    self.assertEqual('quux/bar/baz', wt.id2path(b'baz-id'))