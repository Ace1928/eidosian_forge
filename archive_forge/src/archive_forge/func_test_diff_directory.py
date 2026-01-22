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
def test_diff_directory(self):
    self.build_tree(['new-tree/new-dir/'])
    self.new_tree.add('new-dir', ids=b'new-dir-id')
    self.differ.diff(None, 'new-dir')
    self.assertEqual(self.differ.to_file.getvalue(), b'')