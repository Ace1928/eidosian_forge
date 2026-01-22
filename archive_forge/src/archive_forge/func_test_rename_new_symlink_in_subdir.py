import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_rename_new_symlink_in_subdir(self):
    handler, branch = self.get_handler()
    old_path = b'a/a'
    new_path = b'a/b'
    handler.process(self.get_command_iter(old_path, new_path, 'symlink'))
    revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (new_path,)])