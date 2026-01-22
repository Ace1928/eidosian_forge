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
def create_old_new(self):
    self.build_tree_contents([('old-tree/olddir/',), ('old-tree/olddir/oldfile', b'old\n')])
    self.old_tree.add('olddir')
    self.old_tree.add('olddir/oldfile', ids=b'file-id')
    self.build_tree_contents([('new-tree/newdir/',), ('new-tree/newdir/newfile', b'new\n')])
    self.new_tree.add('newdir')
    self.new_tree.add('newdir/newfile', ids=b'file-id')