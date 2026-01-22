import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_rename_to_deleted_file_in_new_dir(self):
    handler, branch = self.get_handler()
    old_path = b'd1/a'
    new_path = b'd2/b'
    handler.process(self.get_command_iter(old_path, new_path))
    revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'd1',), (old_path,), (b'd2',), (new_path,)])
    revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(b'd1',), (new_path,)], expected_renamed=[(old_path, new_path)])
    self.assertContent(branch, revtree1, old_path, b'aaa')
    self.assertContent(branch, revtree1, new_path, b'bbb')
    self.assertContent(branch, revtree2, new_path, b'aaa')