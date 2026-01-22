import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_criss_cross(self):
    tree_a = self.make_branch_and_tree('a')
    tree_a.commit('', rev_id=b'rev1')
    tree_b = tree_a.controldir.sprout('b').open_workingtree()
    tree_a.commit('', rev_id=b'rev2a')
    tree_b.commit('', rev_id=b'rev2b')
    tree_a.merge_from_branch(tree_b.branch)
    tree_b.merge_from_branch(tree_a.branch)
    tree_a.commit('', rev_id=b'rev3a')
    tree_b.commit('', rev_id=b'rev3b')
    graph = tree_a.branch.repository.get_graph(tree_b.branch.repository)
    out, err = self.run_bzr(['merge', '-d', 'a', 'b'])
    self.assertContainsRe(err, 'Warning: criss-cross merge encountered.')