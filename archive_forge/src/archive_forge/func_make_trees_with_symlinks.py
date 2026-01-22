import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def make_trees_with_symlinks(self):
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    tree2.set_root_id(tree1.path2id(''))
    self.build_tree(['tree1/fromfile', 'tree1/fromdir/'])
    self.build_tree(['tree2/tofile', 'tree2/todir/', 'tree2/unknown'])
    os.symlink('original', 'tree1/changed')
    os.symlink('original', 'tree1/removed')
    os.symlink('original', 'tree1/tofile')
    os.symlink('original', 'tree1/todir')
    os.symlink('unknown', 'tree1/unchanged')
    os.symlink('new', 'tree2/added')
    os.symlink('new', 'tree2/changed')
    os.symlink('new', 'tree2/fromfile')
    os.symlink('new', 'tree2/fromdir')
    os.symlink('unknown', 'tree2/unchanged')
    from_paths_and_ids = ['fromdir', 'fromfile', 'changed', 'removed', 'todir', 'tofile', 'unchanged']
    to_paths_and_ids = ['added', 'fromdir', 'fromfile', 'changed', 'todir', 'tofile', 'unchanged']
    tree1.add(from_paths_and_ids, ids=[p.encode('utf-8') for p in from_paths_and_ids])
    tree2.add(to_paths_and_ids, ids=[p.encode('utf-8') for p in to_paths_and_ids])
    return self.mutable_trees_to_locked_test_trees(tree1, tree2)