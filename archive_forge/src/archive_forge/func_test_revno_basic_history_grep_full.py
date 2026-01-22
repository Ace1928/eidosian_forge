import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_revno_basic_history_grep_full(self):
    """Search for pattern in specific revision number in a file.
        """
    wd = 'foobar0'
    fname = 'file0.txt'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_file(fname, total_lines=0)
    self._mk_versioned_file('file1.txt')
    self._update_file(fname, text='v3 text\n')
    self._update_file(fname, text='v4 text\n')
    self._update_file(fname, text='v5 text\n')
    out, err = self.run_bzr(['grep', '-r', '2', 'v3'])
    self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-r', '3', 'v3'])
    self.assertContainsRe(out, 'file0.txt~3:v3', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-r', '3', '-n', 'v3'])
    self.assertContainsRe(out, 'file0.txt~3:1:v3', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-r', '2', '[tuv]3'])
    self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-r', '3', '[tuv]3'])
    self.assertContainsRe(out, 'file0.txt~3:v3', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-r', '3', '-n', '[tuv]3'])
    self.assertContainsRe(out, 'file0.txt~3:1:v3', flags=TestGrep._reflags)