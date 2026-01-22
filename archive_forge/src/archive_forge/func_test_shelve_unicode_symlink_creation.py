import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_unicode_symlink_creation(self):
    self.requireFeature(features.UnicodeFilenameFeature)
    self._test_shelve_symlink_creation('fo€o', 'b€ar')