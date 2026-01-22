import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_directive_verify_warning(self):
    source = self.make_branch_and_tree('source')
    self.build_tree(['source/a'])
    source.add('a')
    source.commit('Added a', rev_id=b'rev1')
    target = self.make_branch_and_tree('target')
    target.commit('empty commit')
    self.write_directive('directive', source.branch, 'target', b'rev1')
    err = self.run_bzr('merge -d target directive')[1]
    self.assertNotContainsRe(err, 'Preview patch does not match changes')
    target.revert()
    self.write_directive('directive', source.branch, 'target', b'rev1', mangle_patch=True)
    err = self.run_bzr('merge -d target directive')[1]
    self.assertContainsRe(err, 'Preview patch does not match changes')