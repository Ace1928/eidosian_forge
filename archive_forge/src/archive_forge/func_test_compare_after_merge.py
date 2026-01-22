import os
from breezy import conflicts, errors, merge
from breezy.tests import per_workingtree
from breezy.workingtree import PointlessMerge
def test_compare_after_merge(self):
    tree_a = self.make_branch_and_tree('tree_a')
    self.build_tree_contents([('tree_a/file', b'text-a')])
    tree_a.add('file')
    tree_a.commit('added file')
    tree_b = tree_a.controldir.sprout('tree_b').open_workingtree()
    os.unlink('tree_a/file')
    tree_a.commit('deleted file')
    self.build_tree_contents([('tree_b/file', b'text-b')])
    tree_b.commit('changed file')
    tree_a.merge_from_branch(tree_b.branch)
    tree_a.lock_read()
    self.addCleanup(tree_a.unlock)
    list(tree_a.iter_changes(tree_a.basis_tree()))