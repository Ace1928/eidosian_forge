import unittest
import os
from distutils.dep_util import newer, newer_pairwise, newer_group
from distutils.errors import DistutilsFileError
from distutils.tests import support
def test_newer(self):
    tmpdir = self.mkdtemp()
    new_file = os.path.join(tmpdir, 'new')
    old_file = os.path.abspath(__file__)
    self.assertRaises(DistutilsFileError, newer, new_file, old_file)
    self.write_file(new_file)
    self.assertTrue(newer(new_file, 'I_dont_exist'))
    self.assertTrue(newer(new_file, old_file))
    self.assertFalse(newer(old_file, new_file))