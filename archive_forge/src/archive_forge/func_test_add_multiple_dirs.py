import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_add_multiple_dirs(self):
    """Test smart adding multiple directories at once."""
    added_paths = ['file1', 'file2', 'dir1/', 'dir1/file3', 'dir1/subdir2/', 'dir1/subdir2/file4', 'dir2/', 'dir2/file5']
    not_added = ['file6', 'dir3/', 'dir3/file7', 'dir3/file8']
    self.build_tree(added_paths)
    self.build_tree(not_added)
    wt = self.make_branch_and_tree('.')
    wt.smart_add(['file1', 'file2', 'dir1', 'dir2'])
    for path in added_paths:
        self.assertTrue(wt.is_versioned(path.rstrip('/')), 'Failed to add path: {}'.format(path))
    for path in not_added:
        self.assertFalse(wt.is_versioned(path.rstrip('/')), 'Accidentally added path: {}'.format(path))