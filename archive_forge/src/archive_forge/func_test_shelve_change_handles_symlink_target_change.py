import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_change_handles_symlink_target_change(self):
    self._test_shelve_symlink_target_change('foo', 'bar', 'baz', shelve_change=True)