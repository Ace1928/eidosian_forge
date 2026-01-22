from breezy import errors
from breezy.tests.per_branch import TestCaseWithBranch
def test_lookup_dotted_revno(self):
    tree, revmap = self.create_tree_with_merge()
    the_branch = tree.branch
    self.assertEqual((0,), the_branch.revision_id_to_dotted_revno(b'null:'))
    self.assertEqual((1,), the_branch.revision_id_to_dotted_revno(revmap['1']))
    self.assertEqual((2,), the_branch.revision_id_to_dotted_revno(revmap['2']))
    self.assertEqual((3,), the_branch.revision_id_to_dotted_revno(revmap['3']))
    self.assertEqual((1, 1, 1), the_branch.revision_id_to_dotted_revno(revmap['1.1.1']))
    self.assertRaises(errors.NoSuchRevision, the_branch.revision_id_to_dotted_revno, b'rev-1.0.2')