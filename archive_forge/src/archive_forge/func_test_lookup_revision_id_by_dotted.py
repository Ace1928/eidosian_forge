from breezy import errors
from breezy.bzr.fullhistory import FullHistoryBzrBranch
from breezy.tests import TestNotApplicable
from breezy.tests.per_branch import TestCaseWithBranch
def test_lookup_revision_id_by_dotted(self):
    tree, revmap = self.create_tree_with_merge()
    the_branch = tree.branch
    the_branch.lock_read()
    self.addCleanup(the_branch.unlock)
    self.assertEqual(b'null:', the_branch.dotted_revno_to_revision_id((0,)))
    self.assertEqual(revmap['1'], the_branch.dotted_revno_to_revision_id((1,)))
    self.assertEqual(revmap['2'], the_branch.dotted_revno_to_revision_id((2,)))
    self.assertEqual(revmap['3'], the_branch.dotted_revno_to_revision_id((3,)))
    self.assertEqual(revmap['1.1.1'], the_branch.dotted_revno_to_revision_id((1, 1, 1)))
    self.assertRaises(errors.NoSuchRevision, the_branch.dotted_revno_to_revision_id, (1, 0, 2))
    self.assertEqual(None, the_branch._partial_revision_id_to_revno_cache.get(revmap['1']))
    self.assertEqual(revmap['1'], the_branch.dotted_revno_to_revision_id((1,), _cache_reverse=True))
    self.assertEqual((1,), the_branch._partial_revision_id_to_revno_cache.get(revmap['1']))