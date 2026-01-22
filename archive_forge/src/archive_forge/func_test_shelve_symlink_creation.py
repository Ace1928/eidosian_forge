import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_symlink_creation(self):
    self._test_shelve_symlink_creation('foo', 'bar')