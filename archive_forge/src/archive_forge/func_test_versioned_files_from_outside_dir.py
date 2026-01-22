import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_versioned_files_from_outside_dir(self):
    """(versioned) Grep for pattern with dirs passed as argument.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_dir('dir0')
    self._mk_versioned_file('dir0/file0.txt')
    self._mk_versioned_dir('dir1')
    self._mk_versioned_file('dir1/file1.txt')
    out, err = self.run_bzr(['grep', '-r', 'last:1', '.ine1', 'dir0', 'dir1'])
    self.assertContainsRe(out, '^dir0/file0.txt~.:line1', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^dir1/file1.txt~.:line1', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-r', 'last:1', 'line1', 'dir0', 'dir1'])
    self.assertContainsRe(out, '^dir0/file0.txt~.:line1', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^dir1/file1.txt~.:line1', flags=TestGrep._reflags)