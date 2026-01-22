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
def test_internal_diff_no_content(self):
    output = BytesIO()
    diff.internal_diff('old', [], 'new', [], output)
    self.assertEqual(b'', output.getvalue())