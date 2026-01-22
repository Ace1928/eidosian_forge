import gzip
import os
import tarfile
import time
import zipfile
from io import BytesIO
from .. import errors, export, tests
from ..archive.tar import tarball_generator
from ..export import get_root_name
from . import features
def test_export_tarball_generator(self):
    wt = self.make_branch_and_tree('.')
    self.build_tree(['a'])
    wt.add(['a'])
    wt.commit('1', timestamp=42)
    target = BytesIO()
    with wt.lock_read():
        target.writelines(tarball_generator(wt, 'bar'))
    target.seek(0)
    ball2 = tarfile.open(None, 'r', target)
    self.addCleanup(ball2.close)
    self.assertEqual(['bar/a'], ball2.getnames())