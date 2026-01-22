from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_filemodify_path_checking(self):
    self.assertRaises(ValueError, commands.FileModifyCommand, b'', 33188, None, b'text')
    self.assertRaises(ValueError, commands.FileModifyCommand, None, 33188, None, b'text')