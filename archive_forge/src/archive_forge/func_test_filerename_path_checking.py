from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_filerename_path_checking(self):
    self.assertRaises(ValueError, commands.FileRenameCommand, b'', b'foo')
    self.assertRaises(ValueError, commands.FileRenameCommand, None, b'foo')
    self.assertRaises(ValueError, commands.FileRenameCommand, b'foo', b'')
    self.assertRaises(ValueError, commands.FileRenameCommand, b'foo', None)