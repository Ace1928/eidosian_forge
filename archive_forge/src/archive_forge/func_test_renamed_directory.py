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
def test_renamed_directory(self):
    """Test when only a directory is only renamed."""
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/dir/'])
    self.build_tree_contents([('tree/dir/file', b'contents\n')])
    tree.add(['dir', 'dir/file'], ids=[b'dir-id', b'file-id'])
    tree.commit('one', rev_id=b'rev-1')
    tree.rename_one('dir', 'newdir')
    d = get_diff_as_string(tree.basis_tree(), tree)
    self.assertEqual(d, b"=== renamed directory 'dir' => 'newdir'\n")