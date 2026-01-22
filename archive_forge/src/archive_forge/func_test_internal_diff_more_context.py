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
def test_internal_diff_more_context(self):
    output = BytesIO()
    diff.internal_diff('old', [b'same_text\n', b'same_text\n', b'same_text\n', b'same_text\n', b'same_text\n', b'old_text\n'], 'new', [b'same_text\n', b'same_text\n', b'same_text\n', b'same_text\n', b'same_text\n', b'new_text\n'], output, context_lines=4)
    lines = output.getvalue().splitlines(True)
    self.check_patch(lines)
    self.assertEqual([b'--- old\n', b'+++ new\n', b'@@ -2,5 +2,5 @@\n', b' same_text\n', b' same_text\n', b' same_text\n', b' same_text\n', b'-old_text\n', b'+new_text\n', b'\n'], lines)