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
def test_prepare_files(self):
    output = BytesIO()
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/oldname', b'oldcontent')])
    self.build_tree_contents([('tree/oldname2', b'oldcontent2')])
    tree.add('oldname', ids=b'file-id')
    tree.add('oldname2', ids=b'file2-id')
    tree.commit('old tree', timestamp=315532800)
    tree.rename_one('oldname', 'newname')
    tree.rename_one('oldname2', 'newname2')
    self.build_tree_contents([('tree/newname', b'newcontent')])
    self.build_tree_contents([('tree/newname2', b'newcontent2')])
    old_tree = tree.basis_tree()
    old_tree.lock_read()
    self.addCleanup(old_tree.unlock)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    diff_obj = diff.DiffFromTool([sys.executable, '-c', 'print "{old_path} {new_path}"'], old_tree, tree, output)
    self.addCleanup(diff_obj.finish)
    self.assertContainsRe(diff_obj._root, 'brz-diff-[^/]*')
    old_path, new_path = diff_obj._prepare_files('oldname', 'newname')
    self.assertContainsRe(old_path, 'old/oldname$')
    self.assertEqual(315532800, os.stat(old_path).st_mtime)
    self.assertContainsRe(new_path, 'tree/newname$')
    self.assertFileEqual(b'oldcontent', old_path)
    self.assertFileEqual(b'newcontent', new_path)
    if osutils.supports_symlinks(self.test_dir):
        self.assertTrue(os.path.samefile('tree/newname', new_path))
    diff_obj._prepare_files('oldname2', 'newname2')