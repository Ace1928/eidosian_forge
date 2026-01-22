import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_delete_new_file_in_subdir(self):
    handler, branch = self.get_handler()
    path = b'a/a'
    handler.process(self.file_command_iter(path))
    revtree0, revtree1 = self.assertChanges(branch, 1)