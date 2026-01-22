import contextlib
import os.path
import sys
import tempfile
import unittest
from io import open
from os.path import join as pjoin
from ..Dependencies import extended_iglob
def test_extended_iglob_simple_star(self):
    for basedir in 'ad':
        files = [pjoin(basedir, dirname, filename) for dirname in 'xyz' for filename in ['file2_pyx.pyx', 'file2_py.py']]
        self.files_equal(basedir + '/*/*', files)
        self.files_equal(basedir + '/*/*.c12', [])
        self.files_equal(basedir + '/*/*.{py,pyx,c12}', files)
        self.files_equal(basedir + '/*/*.{py,pyx}', files)
        self.files_equal(basedir + '/*/*.{pyx}', files[::2])
        self.files_equal(basedir + '/*/*.pyx', files[::2])
        self.files_equal(basedir + '/*/*.{py}', files[1::2])
        self.files_equal(basedir + '/*/*.py', files[1::2])
        for subdir in 'xy*':
            files = [pjoin(basedir, dirname, filename) for dirname in 'xyz' if subdir in ('*', dirname) for filename in ['file2_pyx.pyx', 'file2_py.py']]
            path = basedir + '/' + subdir + '/'
            self.files_equal(path + '*', files)
            self.files_equal(path + '*.{py,pyx}', files)
            self.files_equal(path + '*.{pyx}', files[::2])
            self.files_equal(path + '*.pyx', files[::2])
            self.files_equal(path + '*.{py}', files[1::2])
            self.files_equal(path + '*.py', files[1::2])