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
def test_nonascii_paths(self):
    self.requireFeature(UnicodeFilenameFeature)
    tt = self.get_empty_tt()
    tt.new_file('ሴfile', tt.root, [b'contents'], b'new-file')
    tt.new_file('other', tt.root, [b'contents'], b'other-file')
    tarfile = self.transform_to_tar(tt)
    tarfile.seek(0)
    tree = self.make_branch_and_tree('bar')
    import_tar(tree, tarfile)
    self.assertPathExists('bar/ሴfile')