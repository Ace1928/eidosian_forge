from breezy.tests import TestNotApplicable
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_get_parent_ids(self):
    t = self.make_branch_and_tree('t1')
    self.assertEqual([], t.get_parent_ids())
    rev1_id = t.commit('foo', allow_pointless=True)
    self.assertEqual([rev1_id], t.get_parent_ids())
    t2 = t.controldir.sprout('t2').open_workingtree()
    rev2_id = t2.commit('foo', allow_pointless=True)
    self.assertEqual([rev2_id], t2.get_parent_ids())
    t.merge_from_branch(t2.branch)
    self.assertEqual([rev1_id, rev2_id], t.get_parent_ids())
    for parent_id in t.get_parent_ids():
        self.assertIsInstance(parent_id, bytes)