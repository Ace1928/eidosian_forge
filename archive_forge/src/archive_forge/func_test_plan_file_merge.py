from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
def test_plan_file_merge(self):
    work_a = self.make_branch_and_tree('wta')
    self.build_tree_contents([('wta/file', b'a\nb\nc\nd\n')])
    work_a.add('file')
    work_a.commit('base version')
    work_b = work_a.controldir.sprout('wtb').open_workingtree()
    self.build_tree_contents([('wta/file', b'b\nc\nd\ne\n')])
    tree_a = self.workingtree_to_test_tree(work_a)
    if getattr(tree_a, 'plan_file_merge', None) is None:
        raise tests.TestNotApplicable('Tree does not support plan_file_merge')
    tree_a.lock_read()
    self.addCleanup(tree_a.unlock)
    self.build_tree_contents([('wtb/file', b'a\nc\nd\nf\n')])
    tree_b = self.workingtree_to_test_tree(work_b)
    tree_b.lock_read()
    self.addCleanup(tree_b.unlock)
    self.assertEqual([('killed-a', b'a\n'), ('killed-b', b'b\n'), ('unchanged', b'c\n'), ('unchanged', b'd\n'), ('new-a', b'e\n'), ('new-b', b'f\n')], list(tree_a.plan_file_merge('file', tree_b)))