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
def test_diff_text(self):
    self.build_tree_contents([('old-tree/olddir/',), ('old-tree/olddir/oldfile', b'old\n')])
    self.old_tree.add('olddir')
    self.old_tree.add('olddir/oldfile', ids=b'file-id')
    self.build_tree_contents([('new-tree/newdir/',), ('new-tree/newdir/newfile', b'new\n')])
    self.new_tree.add('newdir')
    self.new_tree.add('newdir/newfile', ids=b'file-id')
    differ = diff.DiffText(self.old_tree, self.new_tree, BytesIO())
    differ.diff_text('olddir/oldfile', None, 'old label', 'new label')
    self.assertEqual(b'--- old label\n+++ new label\n@@ -1,1 +0,0 @@\n-old\n\n', differ.to_file.getvalue())
    differ.to_file.seek(0)
    differ.diff_text(None, 'newdir/newfile', 'old label', 'new label')
    self.assertEqual(b'--- old label\n+++ new label\n@@ -0,0 +1,1 @@\n+new\n\n', differ.to_file.getvalue())
    differ.to_file.seek(0)
    differ.diff_text('olddir/oldfile', 'newdir/newfile', 'old label', 'new label')
    self.assertEqual(b'--- old label\n+++ new label\n@@ -1,1 +1,1 @@\n-old\n+new\n\n', differ.to_file.getvalue())