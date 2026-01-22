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
def test_tgz(self):
    wt = self.make_branch_and_tree('.')
    self.build_tree(['a'])
    wt.add(['a'])
    wt.commit('1')
    export.export(wt, 'target.tar.gz', format='tgz')
    tf = tarfile.open('target.tar.gz')
    self.assertEqual(['target/a'], tf.getnames())