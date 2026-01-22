import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_revno_versioned_file_in_dir(self):
    """Grep specific version of file withing dir.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_dir('dir0')
    self._mk_versioned_file('dir0/file0.txt')
    self._update_file('dir0/file0.txt', 'v3 text\n')
    self._update_file('dir0/file0.txt', 'v4 text\n')
    self._update_file('dir0/file0.txt', 'v5 text\n')
    out, err = self.run_bzr(['grep', '-r', 'last:3', 'v4'])
    self.assertNotContainsRe(out, '^dir0/file0.txt', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-r', 'last:2', 'v4'])
    self.assertContainsRe(out, '^dir0/file0.txt~4:v4', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-r', 'last:3', '[tuv]4'])
    self.assertNotContainsRe(out, '^dir0/file0.txt', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-r', 'last:2', '[tuv]4'])
    self.assertContainsRe(out, '^dir0/file0.txt~4:v4', flags=TestGrep._reflags)