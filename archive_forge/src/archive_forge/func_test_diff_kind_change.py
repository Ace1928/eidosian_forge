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
def test_diff_kind_change(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    self.build_tree_contents([('old-tree/olddir/',), ('old-tree/olddir/oldfile', b'old\n')])
    self.old_tree.add('olddir')
    self.old_tree.add('olddir/oldfile', ids=b'file-id')
    self.build_tree(['new-tree/newdir/'])
    os.symlink('new', 'new-tree/newdir/newfile')
    self.new_tree.add('newdir')
    self.new_tree.add('newdir/newfile', ids=b'file-id')
    self.differ.diff('olddir/oldfile', 'newdir/newfile')
    self.assertContainsRe(self.differ.to_file.getvalue(), b'--- olddir/oldfile.*\\n\\+\\+\\+ newdir/newfile.*\\n\\@\\@ -1,1 \\+0,0 \\@\\@\\n-old\\n\\n')
    self.assertContainsRe(self.differ.to_file.getvalue(), b"=== target is 'new'\n")