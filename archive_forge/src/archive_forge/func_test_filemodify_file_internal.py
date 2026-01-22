from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_filemodify_file_internal(self):
    c = commands.FileModifyCommand(b'foo/bar', 33188, None, b'hello world')
    self.assertEqual(b'M 644 inline foo/bar\ndata 11\nhello world', bytes(c))