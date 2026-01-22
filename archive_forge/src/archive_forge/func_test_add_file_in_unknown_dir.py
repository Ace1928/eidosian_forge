import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_add_file_in_unknown_dir(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['dir/', 'dir/subdir/', 'dir/subdir/foo'])
    tree.smart_add(['dir/subdir/foo'])
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual(['', 'dir', 'dir/subdir', 'dir/subdir/foo'], [path for path, ie in tree.iter_entries_by_dir()])