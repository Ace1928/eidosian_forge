from breezy import errors
from breezy.tests import TestNotApplicable
from breezy.tests.per_branch import TestCaseWithBranch
def test_simple_revno(self):
    tree, revmap = self.create_tree_with_merge()
    the_branch = tree.branch
    self.assertEqual(0, the_branch.revision_id_to_revno(b'null:'))
    self.assertEqual(1, the_branch.revision_id_to_revno(revmap['1']))
    self.assertEqual(2, the_branch.revision_id_to_revno(revmap['2']))
    self.assertEqual(3, the_branch.revision_id_to_revno(revmap['3']))
    self.assertRaises(errors.NoSuchRevision, the_branch.revision_id_to_revno, b'rev-none')
    self.assertRaises(errors.NoSuchRevision, the_branch.revision_id_to_revno, revmap['1.1.1'])