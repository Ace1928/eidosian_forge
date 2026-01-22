import pathlib
from tempfile import gettempdir
import unittest
from traits.api import BaseDirectory, Directory, HasTraits, TraitError
def test_accepts_valid_dir_name(self):
    foo = ExistsBaseDirectory()
    tempdir = gettempdir()
    self.assertIsInstance(tempdir, str)
    foo.path = tempdir