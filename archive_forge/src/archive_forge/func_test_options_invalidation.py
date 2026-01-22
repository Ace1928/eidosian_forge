import difflib
import glob
import gzip
import os
import sys
import tempfile
import unittest
import Cython.Build.Dependencies
import Cython.Utils
from Cython.TestUtils import CythonTest
def test_options_invalidation(self):
    hash_pyx = os.path.join(self.src_dir, 'options.pyx')
    hash_c = hash_pyx[:-len('.pyx')] + '.c'
    with open(hash_pyx, 'w') as f:
        f.write('pass')
    self.fresh_cythonize(hash_pyx, cache=self.cache_dir, cplus=False)
    self.assertEqual(1, len(self.cache_files('options.c*')))
    os.unlink(hash_c)
    self.fresh_cythonize(hash_pyx, cache=self.cache_dir, cplus=True)
    self.assertEqual(2, len(self.cache_files('options.c*')))
    os.unlink(hash_c)
    self.fresh_cythonize(hash_pyx, cache=self.cache_dir, cplus=False, show_version=False)
    self.assertEqual(2, len(self.cache_files('options.c*')))
    os.unlink(hash_c)
    self.fresh_cythonize(hash_pyx, cache=self.cache_dir, cplus=False, show_version=True)
    self.assertEqual(2, len(self.cache_files('options.c*')))