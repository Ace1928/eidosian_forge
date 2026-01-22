import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_missing_second_revision_spec(self):
    """Merge uses branch basis when the second revision is unspecified."""
    this = self.make_branch_and_tree('this')
    this.commit('rev1')
    other = self.make_branch_and_tree('other')
    self.build_tree(['other/other_file'])
    other.add('other_file')
    other.commit('rev1b')
    self.run_bzr('merge -d this other -r0..')
    self.assertPathExists('this/other_file')