import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_tgz_export(self):
    self.run_tar_export_disk_and_stdout('tgz', 'gz')