import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_zip_export_per_file_timestamps(self):
    tree = self.example_branch()
    self.build_tree_contents([('branch/har', b'foo')])
    tree.add('har')
    timestamp = 347151600
    tree.commit('setup', timestamp=timestamp)
    self.run_bzr('export --per-file-timestamps test.zip branch')
    zfile = zipfile.ZipFile('test.zip')
    info = zfile.getinfo('test/har')
    self.assertEqual(time.localtime(timestamp)[:6], info.date_time)