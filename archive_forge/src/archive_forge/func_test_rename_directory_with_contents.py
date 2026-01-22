import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_rename_directory_with_contents(self):
    state = self.assertUpdate(active=[('dir1/', b'dir-id'), ('dir1/file', b'file-id')], basis=[('dir1/', b'dir-id'), ('dir1/file', b'file-id')], target=[('dir2/', b'dir-id'), ('dir2/file', b'file-id')])
    state = self.assertUpdate(active=[('dir2/', b'dir-id'), ('dir2/file', b'file-id')], basis=[('dir1/', b'dir-id'), ('dir1/file', b'file-id')], target=[('dir2/', b'dir-id'), ('dir2/file', b'file-id')])
    state = self.assertUpdate(active=[], basis=[('dir1/', b'dir-id'), ('dir1/file', b'file-id')], target=[('dir2/', b'dir-id'), ('dir2/file', b'file-id')])
    state = self.assertUpdate(active=[('dir3/', b'dir-id'), ('dir3/file', b'file-id')], basis=[('dir1/', b'dir-id'), ('dir1/file', b'file-id')], target=[('dir2/', b'dir-id'), ('dir2/file', b'file-id')])
    state = self.assertUpdate(active=[('dir1/', b'dir1-id'), ('dir1/file', b'file1-id'), ('dir2/', b'dir2-id'), ('dir2/file', b'file2-id')], basis=[('dir1/', b'dir-id'), ('dir1/file', b'file-id')], target=[('dir2/', b'dir-id'), ('dir2/file', b'file-id')])