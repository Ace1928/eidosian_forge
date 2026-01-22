from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_filedelete(self):
    c = commands.FileDeleteCommand(b'foo/bar')
    self.assertEqual(b'D foo/bar', bytes(c))