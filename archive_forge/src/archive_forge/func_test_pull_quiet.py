import os
import sys
from breezy import (branch, debug, osutils, tests, uncommit, urlutils,
from breezy.bzr import remote
from breezy.directory_service import directories
from breezy.tests import fixtures, script
def test_pull_quiet(self):
    """Check that brz pull --quiet does not print anything"""
    tree_a = self.make_branch_and_tree('tree_a')
    self.build_tree(['tree_a/foo'])
    tree_a.add('foo')
    revision_id = tree_a.commit('bar')
    tree_b = tree_a.controldir.sprout('tree_b').open_workingtree()
    out, err = self.run_bzr('pull --quiet -d tree_b')
    self.assertEqual(out, '')
    self.assertEqual(err, '')
    self.assertEqual(tree_b.last_revision(), revision_id)
    self.build_tree(['tree_a/moo'])
    tree_a.add('moo')
    revision_id = tree_a.commit('quack')
    out, err = self.run_bzr('pull --quiet -d tree_b')
    self.assertEqual(out, '')
    self.assertEqual(err, '')
    self.assertEqual(tree_b.last_revision(), revision_id)