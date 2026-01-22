from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_filerename_quoted(self):
    c = commands.FileRenameCommand(b'foo/b a r', b'foo/b a z')
    self.assertEqual(b'R "foo/b a r" foo/b a z', bytes(c))