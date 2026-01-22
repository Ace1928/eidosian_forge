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
def test_modified_file(self):
    """Test when a file is modified."""
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/file', b'contents\n')])
    tree.add(['file'], ids=[b'file-id'])
    tree.commit('one', rev_id=b'rev-1')
    self.build_tree_contents([('tree/file', b'new contents\n')])
    d = get_diff_as_string(tree.basis_tree(), tree)
    self.assertContainsRe(d, b"=== modified file 'file'\n")
    self.assertContainsRe(d, b'--- old/file\t')
    self.assertContainsRe(d, b'\\+\\+\\+ new/file\t')
    self.assertContainsRe(d, b'-contents\n\\+new contents\n')