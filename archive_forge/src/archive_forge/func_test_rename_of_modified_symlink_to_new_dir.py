import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_rename_of_modified_symlink_to_new_dir(self):
    handler, branch = self.get_handler()
    old_path = b'd1/a'
    new_path = b'd2/b'
    handler.process(self.get_command_iter(old_path, new_path, 'symlink'))
    revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'd1',), (old_path,)])
    revtree1, revtree2 = self.assertChanges(branch, 2, expected_renamed=[(old_path, new_path)], expected_added=[(b'd2',)], expected_removed=[(b'd1',)])
    self.assertSymlinkTarget(branch, revtree1, old_path, 'aaa')
    self.assertSymlinkTarget(branch, revtree2, new_path, 'bbb')