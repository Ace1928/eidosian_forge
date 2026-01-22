import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_zip_export_ignores_bzr(self):
    tree = self.make_tree_with_extra_bzr_files()
    self.assertTrue(tree.has_filename('.bzrignore'))
    self.assertTrue(tree.has_filename('.bzrrules'))
    self.assertTrue(tree.has_filename('.bzr-adir'))
    self.assertTrue(tree.has_filename('.bzr-adir/afile'))
    self.run_bzr('export test.zip -d tree')
    zfile = zipfile.ZipFile('test.zip')
    self.assertEqual(['test/a'], sorted(zfile.namelist()))