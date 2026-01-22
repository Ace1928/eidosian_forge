import contextlib
import os
import re
import subprocess
import sys
import tempfile
from io import BytesIO
from .. import diff, errors, osutils
from .. import revision as _mod_revision
from .. import revisionspec, revisiontree, tests
from ..tests import EncodingAdapter, features
from ..tests.scenarios import load_tests_apply_scenarios
def test_binary_unicode_filenames(self):
    """Test that contents of files are *not* encoded in UTF-8 when there
        is a binary file in the diff.
        """
    self.requireFeature(features.UnicodeFilenameFeature)
    tree = self.make_branch_and_tree('tree')
    alpha, omega = ('α', 'ω')
    alpha_utf8, omega_utf8 = (alpha.encode('utf8'), omega.encode('utf8'))
    self.build_tree_contents([('tree/' + alpha, b'\x00'), ('tree/' + omega, b'The %s and the %s\n' % (alpha_utf8, omega_utf8))])
    tree.add([alpha])
    tree.add([omega])
    diff_content = StubO()
    diff.show_diff_trees(tree.basis_tree(), tree, diff_content)
    diff_content.check_types(self, bytes)
    d = b''.join(diff_content.write_record)
    self.assertContainsRe(d, b"=== added file '%s'" % alpha_utf8)
    self.assertContainsRe(d, b'Binary files a/%s.*and b/%s.* differ\n' % (alpha_utf8, alpha_utf8))
    self.assertContainsRe(d, b"=== added file '%s'" % omega_utf8)
    self.assertContainsRe(d, b'--- a/%s' % (omega_utf8,))
    self.assertContainsRe(d, b'\\+\\+\\+ b/%s' % (omega_utf8,))