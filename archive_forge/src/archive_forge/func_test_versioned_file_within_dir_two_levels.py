import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_versioned_file_within_dir_two_levels(self):
    """(versioned) Search for pattern while in nested dir (two levels).
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_dir('dir0')
    self._mk_versioned_dir('dir0/dir1')
    self._mk_versioned_file('dir0/dir1/file0.txt')
    os.chdir('dir0')
    out, err = self.run_bzr(['grep', '-r', 'last:1', '.ine1'])
    self.assertContainsRe(out, '^dir1/file0.txt~.:line1', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-r', 'last:1', '--from-root', 'l.ne1'])
    self.assertContainsRe(out, '^dir0/dir1/file0.txt~.:line1', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-r', 'last:1', '--no-recursive', 'line1'])
    self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-r', 'last:1', 'lin.1'])
    self.assertContainsRe(out, '^dir1/file0.txt~.:line1', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-r', 'last:1', '--from-root', 'line1'])
    self.assertContainsRe(out, '^dir0/dir1/file0.txt~.:line1', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-r', 'last:1', '--no-recursive', 'line1'])
    self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 0)