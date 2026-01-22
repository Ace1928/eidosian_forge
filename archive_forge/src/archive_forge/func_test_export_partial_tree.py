import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_export_partial_tree(self):
    tree = self.example_branch()
    self.build_tree(['branch/subdir/', 'branch/subdir/foo.txt'])
    tree.smart_add(['branch'])
    tree.commit('more setup')
    out, err = self.run_bzr('export exported branch/subdir')
    self.assertEqual(['foo.txt'], os.listdir('exported'))