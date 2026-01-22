import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def assertZipANameAndContent(self, zfile, root=''):
    """The file should only contain name 'a' and _file_content"""
    fname = root + 'a'
    self.assertEqual([fname], sorted(zfile.namelist()))
    zfile.testzip()
    self.assertEqualDiff(self._file_content, zfile.read(fname))