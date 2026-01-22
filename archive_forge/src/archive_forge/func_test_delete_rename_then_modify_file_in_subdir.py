import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_delete_rename_then_modify_file_in_subdir(self):
    handler, branch = self.get_handler()
    old_path = b'd/a'
    new_path = b'd/b'
    handler.process(self.get_command_iter(old_path, new_path))
    revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'd',), (old_path,), (new_path,)])
    revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(new_path,)], expected_renamed=[(old_path, new_path)])
    self.assertContent(branch, revtree1, old_path, b'aaa')
    self.assertContent(branch, revtree1, new_path, b'zzz')
    self.assertContent(branch, revtree2, new_path, b'bbb')