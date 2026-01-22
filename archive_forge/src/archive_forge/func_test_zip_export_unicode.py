import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_zip_export_unicode(self):
    self.requireFeature(features.UnicodeFilenameFeature)
    tree = self.make_branch_and_tree('zip')
    fname = '€.txt'
    self.build_tree(['zip/' + fname])
    tree.add([fname])
    tree.commit('first')
    os.chdir('zip')
    self.run_bzr('export test.zip')
    zfile = zipfile.ZipFile('test.zip')
    self.assertEqual(['test/' + fname], sorted(zfile.namelist()))