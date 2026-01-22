import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_directive_cherrypick(self):
    source = self.make_branch_and_tree('source')
    source.commit('nothing')
    target = source.controldir.sprout('target').open_workingtree()
    self.build_tree(['source/a'])
    source.add('a')
    source.commit('Added a', rev_id=b'rev1')
    self.build_tree(['source/b'])
    source.add('b')
    source.commit('Added b', rev_id=b'rev2')
    target.commit('empty commit')
    self.write_directive('directive', source.branch, 'target', b'rev2', b'rev1')
    out, err = self.run_bzr('merge -d target directive')
    self.assertPathDoesNotExist('target/a')
    self.assertPathExists('target/b')
    self.assertContainsRe(err, 'Performing cherrypick')