import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_delete_file_single_level(self):
    handler, branch = self.get_handler()
    paths = [b'a/b/c', b'a/b/d/e']
    paths_to_delete = [b'a/b/d/e']
    handler.process(self.file_command_iter(paths, paths_to_delete))
    revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (b'a/b',), (b'a/b/c',), (b'a/b/d',), (b'a/b/d/e',)])
    revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(b'a/b/d',), (b'a/b/d/e',)])