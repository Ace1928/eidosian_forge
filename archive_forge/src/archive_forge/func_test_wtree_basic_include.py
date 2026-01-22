import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_wtree_basic_include(self):
    """(wtree) Ensure that --include flag is respected.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_file('file0.aa')
    self._mk_versioned_file('file0.bb')
    self._mk_versioned_file('file0.cc')
    out, err = self.run_bzr(['grep', '--include', '*.aa', '--include', '*.bb', 'line1'])
    self.assertContainsRe(out, 'file0.aa:line1', flags=TestGrep._reflags)
    self.assertContainsRe(out, 'file0.bb:line1', flags=TestGrep._reflags)
    self.assertNotContainsRe(out, 'file0.cc', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 4)
    out, err = self.run_bzr(['grep', '--include', '*.aa', '--include', '*.bb', 'line1$'])
    self.assertContainsRe(out, 'file0.aa:line1', flags=TestGrep._reflags)
    self.assertContainsRe(out, 'file0.bb:line1', flags=TestGrep._reflags)
    self.assertNotContainsRe(out, 'file0.cc', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 2)