import shutil
import tempfile
from ..lfs import LFSStore
from . import TestCase
def test_missing(self):
    self.assertRaises(KeyError, self.lfs.open_object, 'abcdeabcdeabcdeabcde')