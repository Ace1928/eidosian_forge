import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_delete_dir(self):
    handler, branch = self.get_handler()
    paths = [b'a/b/c', b'a/b/d', b'a/b/e/f', b'a/g']
    dir = b'a/b'
    handler.process(self.file_command_iter(paths, dir))
    revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (b'a/b',), (b'a/b/c',), (b'a/b/d',), (b'a/b/e',), (b'a/b/e/f',), (b'a/g',)])
    revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(b'a/b',), (b'a/b/c',), (b'a/b/d',), (b'a/b/e',), (b'a/b/e/f',)])