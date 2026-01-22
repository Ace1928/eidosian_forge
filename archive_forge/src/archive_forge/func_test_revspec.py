import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_revspec(self):
    """Ensure various revspecs work
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_dir('dir0')
    self._mk_versioned_file('dir0/file0.txt')
    self._update_file('dir0/file0.txt', 'v3 text\n')
    self._update_file('dir0/file0.txt', 'v4 text\n')
    self._update_file('dir0/file0.txt', 'v5 text\n')
    out, err = self.run_bzr(['grep', '-r', 'revno:1..2', 'v3'])
    self.assertNotContainsRe(out, 'file0', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 0)
    out, err = self.run_bzr(['grep', '-r', 'revno:4..', 'v4'])
    self.assertContainsRe(out, '^dir0/file0.txt', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 2)
    out, err = self.run_bzr(['grep', '-r', '..revno:3', 'v4'])
    self.assertNotContainsRe(out, 'file0', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 0)
    out, err = self.run_bzr(['grep', '-r', '..revno:3', 'v3'])
    self.assertContainsRe(out, '^dir0/file0.txt', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 1)