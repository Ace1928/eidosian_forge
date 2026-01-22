import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_rename_symlink_becomes_directory(self):
    handler, branch = self.get_handler()
    old_path2 = b'foo'
    path1 = b'a/b'
    new_path2 = b'a/b/c'
    handler.process(self.file_command_iter(path1, old_path2, new_path2, 'symlink'))
    revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (path1,), (old_path2,)])
    revtree1, revtree2 = self.assertChanges(branch, 2, expected_renamed=[(old_path2, new_path2)], expected_kind_changed=[(path1, 'symlink', 'directory')])
    self.assertSymlinkTarget(branch, revtree1, path1, 'aaa')
    self.assertSymlinkTarget(branch, revtree2, new_path2, 'bbb')