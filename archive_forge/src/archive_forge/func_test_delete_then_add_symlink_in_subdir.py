import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_delete_then_add_symlink_in_subdir(self):
    handler, branch = self.get_handler()
    path = b'a/a'
    handler.process(self.file_command_iter(path, kind='symlink'))
    revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (path,)])
    revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(path,)], expected_added=[(path,)])
    self.assertSymlinkTarget(branch, revtree1, path, 'aaa')
    self.assertSymlinkTarget(branch, revtree2, path, 'bbb')