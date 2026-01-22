from __future__ import unicode_literals
import errno
import posixpath
from unittest import TestCase
from pybtex import io
def test_open_missing(self):
    self.assertRaises(EnvironmentError, io._open_existing, self.fs.open, 'nosuchfile.bbl', 'rb', locate=self.fs.locate)