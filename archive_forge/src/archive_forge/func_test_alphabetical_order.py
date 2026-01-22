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
def test_alphabetical_order(self):
    self.build_tree(['new-tree/a-file'])
    self.new_tree.add('a-file')
    self.build_tree(['old-tree/b-file'])
    self.old_tree.add('b-file')
    self.differ.show_diff(None)
    self.assertContainsRe(self.differ.to_file.getvalue(), b'.*a-file(.|\n)*b-file')