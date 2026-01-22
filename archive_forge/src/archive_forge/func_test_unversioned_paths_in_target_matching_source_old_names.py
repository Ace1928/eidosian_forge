import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_unversioned_paths_in_target_matching_source_old_names(self):
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    tree2.set_root_id(tree1.path2id(''))
    self.build_tree(['tree2/file', 'tree2/dir/', 'tree1/file', 'tree2/movedfile', 'tree1/dir/', 'tree2/moveddir/'])
    if supports_symlinks(self.test_dir):
        os.symlink('target', 'tree1/link')
        os.symlink('target', 'tree2/link')
        os.symlink('target', 'tree2/movedlink')
        links_supported = True
    else:
        links_supported = False
    tree1.add(['file', 'dir'], ids=[b'file-id', b'dir-id'])
    tree2.add(['movedfile', 'moveddir'], ids=[b'file-id', b'dir-id'])
    if links_supported:
        tree1.add(['link'], ids=[b'link-id'])
        tree2.add(['movedlink'], ids=[b'link-id'])
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    self.not_applicable_if_cannot_represent_unversioned(tree2)
    root_id = tree1.path2id('')
    expected = [self.renamed(tree1, tree2, 'dir', 'moveddir', False), self.renamed(tree1, tree2, 'file', 'movedfile', True), self.unversioned(tree2, 'file'), self.unversioned(tree2, 'dir')]
    specific_files = ['file', 'dir']
    if links_supported:
        expected.append(self.renamed(tree1, tree2, 'link', 'movedlink', False))
        expected.append(self.unversioned(tree2, 'link'))
        specific_files.append('link')
    expected = self.sorted(expected)
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2, require_versioned=False, want_unversioned=True))
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2, specific_files=specific_files, require_versioned=False, want_unversioned=True))