import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_multiple_wtree_files(self):
    """(wtree) Search for pattern in multiple files in working tree.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_file('file0.txt', total_lines=2)
    self._mk_versioned_file('file1.txt', total_lines=2)
    self._mk_versioned_file('file2.txt', total_lines=2)
    self._update_file('file0.txt', 'HELLO\n', checkin=False)
    self._update_file('file1.txt', 'HELLO\n', checkin=True)
    self._update_file('file2.txt', 'HELLO\n', checkin=False)
    out, err = self.run_bzr(['grep', 'HELLO', 'file0.txt', 'file1.txt', 'file2.txt'])
    self.assertContainsRe(out, 'file0.txt:HELLO', flags=TestGrep._reflags)
    self.assertContainsRe(out, 'file1.txt:HELLO', flags=TestGrep._reflags)
    self.assertContainsRe(out, 'file2.txt:HELLO', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 3)
    out, err = self.run_bzr(['grep', 'HELLO', '-r', 'last:1', 'file0.txt', 'file1.txt', 'file2.txt'])
    self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
    self.assertContainsRe(out, 'file1.txt~.:HELLO', flags=TestGrep._reflags)
    self.assertNotContainsRe(out, 'file2.txt', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 1)
    out, err = self.run_bzr(['grep', 'HE..O', 'file0.txt', 'file1.txt', 'file2.txt'])
    self.assertContainsRe(out, 'file0.txt:HELLO', flags=TestGrep._reflags)
    self.assertContainsRe(out, 'file1.txt:HELLO', flags=TestGrep._reflags)
    self.assertContainsRe(out, 'file2.txt:HELLO', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 3)
    out, err = self.run_bzr(['grep', 'HE..O', '-r', 'last:1', 'file0.txt', 'file1.txt', 'file2.txt'])
    self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
    self.assertContainsRe(out, 'file1.txt~.:HELLO', flags=TestGrep._reflags)
    self.assertNotContainsRe(out, 'file2.txt', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 1)