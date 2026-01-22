import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_zip_export_stdout(self):
    tree = self.make_basic_tree()
    contents = self.run_bzr_raw('export -d tree --format=zip -')[0]
    self.assertZipANameAndContent(zipfile.ZipFile(BytesIO(contents)))