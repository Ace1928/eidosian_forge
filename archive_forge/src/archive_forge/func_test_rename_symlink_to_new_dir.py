import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_rename_symlink_to_new_dir(self):
    handler, branch = self.get_handler()
    old_path = b'a/a'
    new_path = b'b/a'
    handler.process(self.get_command_iter(old_path, new_path, 'symlink'))
    self.assertChanges(branch, 2, expected_renamed=[(old_path, new_path)], expected_added=[(b'b',)], expected_removed=[(b'a',)])