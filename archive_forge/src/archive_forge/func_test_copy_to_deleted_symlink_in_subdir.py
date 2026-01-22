import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_copy_to_deleted_symlink_in_subdir(self):
    handler, branch = self.get_handler()
    src_path = b'd/a'
    dest_path = b'd/b'
    handler.process(self.file_command_iter(src_path, dest_path, 'symlink'))
    revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'd',), (src_path,), (dest_path,)])
    revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(dest_path,)], expected_added=[(dest_path,)])
    self.assertSymlinkTarget(branch, revtree1, src_path, 'aaa')
    self.assertSymlinkTarget(branch, revtree1, dest_path, 'bbb')
    self.assertSymlinkTarget(branch, revtree2, src_path, 'aaa')
    self.assertSymlinkTarget(branch, revtree2, dest_path, 'aaa')