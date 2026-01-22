import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_explicit_reprocess_show_base(self):
    tree, other = self.create_conflicting_branches()
    self.run_bzr_error(['Cannot do conflict reduction and show base'], 'merge ../other --reprocess --show-base', working_dir='tree')