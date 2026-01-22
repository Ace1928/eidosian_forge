import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_delete_symlink_in_root(self):
    handler, branch = self.get_handler()
    path = b'a'
    handler.process(self.file_command_iter(path, kind='symlink'))
    revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(path,)])
    self.assertSymlinkTarget(branch, revtree1, path, 'aaa')
    self.assertRevisionRoot(revtree1, path)