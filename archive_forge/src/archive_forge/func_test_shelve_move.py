import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_move(self):
    creator, tree = self.prepare_shelve_move()
    creator.shelve_rename(b'baz-id')
    self.check_shelve_move(creator, tree)