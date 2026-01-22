import os
import tarfile
import tempfile
import warnings
from io import BytesIO
from shutil import copy2, copytree, rmtree
from .. import osutils
from .. import revision as _mod_revision
from .. import transform
from ..controldir import ControlDir
from ..export import export
from ..upstream_import import (NotArchiveType, ZipFileWrapper,
from . import TestCaseInTempDir, TestCaseWithTransport
from .features import UnicodeFilenameFeature
def test_get_archive_type(self):
    self.assertEqual(('tar', None), get_archive_type('foo.tar'))
    self.assertEqual(('zip', None), get_archive_type('foo.zip'))
    self.assertRaises(NotArchiveType, get_archive_type, 'foo.gif')
    self.assertEqual(('tar', 'gz'), get_archive_type('foo.tar.gz'))
    self.assertRaises(NotArchiveType, get_archive_type, 'foo.zip.gz')
    self.assertEqual(('tar', 'gz'), get_archive_type('foo.tgz'))
    self.assertEqual(('tar', 'lzma'), get_archive_type('foo.tar.lzma'))
    self.assertEqual(('tar', 'lzma'), get_archive_type('foo.tar.xz'))
    self.assertEqual(('tar', 'bz2'), get_archive_type('foo.tar.bz2'))