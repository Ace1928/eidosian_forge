import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_wtree_file_in_dir_no_recursive(self):
    """(wtree) Should not recurse with --no-recursive"""
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_file('fileX.txt', line_prefix='lin')
    self._mk_versioned_dir('dir0')
    self._mk_versioned_file('dir0/file0.txt')
    out, err = self.run_bzr(['grep', '--no-recursive', 'line1'])
    self.assertNotContainsRe(out, 'file0.txt:line1', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 0)
    out, err = self.run_bzr(['grep', '--no-recursive', 'lin.1'])
    self.assertNotContainsRe(out, 'file0.txt:line1', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 0)