import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_add_above_tree_preserves_tree(self):
    """Test nested trees are not affect by an add above them."""
    paths = ('original/', 'original/file1', 'original/file2')
    child_paths = ('path',)
    full_child_paths = ('original/child', 'original/child/path')
    build_paths = ('original/', 'original/file1', 'original/file2', 'original/child/', 'original/child/path')
    self.build_tree(build_paths)
    wt = self.make_branch_and_tree('.')
    if wt.controldir.user_url != wt.branch.controldir.user_url:
        wt.branch.controldir.root_transport.mkdir('original')
    child_tree = self.make_branch_and_tree('original/child')
    wt.smart_add(('.',))
    for path in paths:
        self.assertNotEqual((path, wt.is_versioned(path)), (path, False))
    for path in full_child_paths:
        self.assertEqual((path, wt.is_versioned(path)), (path, False))
    for path in child_paths:
        self.assertFalse(child_tree.is_versioned(path))