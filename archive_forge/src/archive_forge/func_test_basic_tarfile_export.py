import os
import stat
import tarfile
import time
import zipfile
from io import BytesIO
from ... import export, osutils
from ...archive import zip
from .. import TestCaseWithTransport, features
def test_basic_tarfile_export(self):
    self.example_branch()
    os.chdir('branch')
    self.run_bzr('export ../first.tar -r 1')
    self.assertTrue(os.path.isfile('../first.tar'))
    with tarfile.open('../first.tar') as tf:
        self.assertEqual(['first/hello'], sorted(tf.getnames()))
        self.assertEqual(b'foo', tf.extractfile('first/hello').read())
    self.run_bzr('export ../first.tar.gz -r 1')
    self.assertTrue(os.path.isfile('../first.tar.gz'))
    self.run_bzr('export ../first.tbz2 -r 1')
    self.assertTrue(os.path.isfile('../first.tbz2'))
    self.run_bzr('export ../first.tar.bz2 -r 1')
    self.assertTrue(os.path.isfile('../first.tar.bz2'))
    self.run_bzr('export ../first.tar.tbz2 -r 1')
    self.assertTrue(os.path.isfile('../first.tar.tbz2'))
    with tarfile.open('../first.tar.tbz2', 'r:bz2') as tf:
        self.assertEqual(['first.tar/hello'], sorted(tf.getnames()))
        self.assertEqual(b'foo', tf.extractfile('first.tar/hello').read())
    self.run_bzr('export ../first2.tar -r 1 --root pizza')
    with tarfile.open('../first2.tar') as tf:
        self.assertEqual(['pizza/hello'], sorted(tf.getnames()))
        self.assertEqual(b'foo', tf.extractfile('pizza/hello').read())