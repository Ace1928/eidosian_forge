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
def test_lzma(self):
    self.requireFeature(features.lzma)
    import lzma
    wt = self.make_branch_and_tree('.')
    self.build_tree(['a'])
    wt.add(['a'])
    wt.commit('1')
    export.export(wt, 'target.tar.lzma', format='tlzma')
    tf = tarfile.open(fileobj=lzma.LZMAFile('target.tar.lzma'))
    self.assertEqual(['target/a'], tf.getnames())