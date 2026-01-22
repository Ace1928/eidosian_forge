import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_versioned_ignore_case_match(self):
    """(versioned) Match fails without --ignore-case.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_file('file0.txt')
    out, err = self.run_bzr(['grep', '-r', 'last:1', '-i', 'Li.E1', 'file0.txt'])
    self.assertContainsRe(out, 'file0.txt~.:line1', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-r', 'last:1', '-i', 'LinE1', 'file0.txt'])
    self.assertContainsRe(out, 'file0.txt~.:line1', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-r', 'last:1', '--ignore-case', 'LinE1', 'file0.txt'])
    self.assertContainsRe(out, '^file0.txt~.:line1', flags=TestGrep._reflags)