import pathlib
from tempfile import gettempdir
import unittest
from traits.api import BaseDirectory, Directory, HasTraits, TraitError
def test_simple_accepts_any_name(self):
    """ BaseDirectory with no existence check accepts any path name.
        """
    foo = SimpleBaseDirectory()
    foo.path = '!!!invalid_directory'