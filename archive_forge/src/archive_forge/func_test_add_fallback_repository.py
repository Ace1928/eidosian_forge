from breezy import errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable
from breezy.tests.per_repository import TestCaseWithRepository
def test_add_fallback_repository(self):
    repo = self.make_repository('repo')
    tree = self.make_branch_and_tree('branch')
    if not repo._format.supports_external_lookups:
        self.assertRaises(errors.UnstackableRepositoryFormat, repo.add_fallback_repository, tree.branch.repository)
        raise TestNotApplicable
    repo.add_fallback_repository(tree.branch.repository)
    revision_id = tree.commit('1st post')
    repo.lock_read()
    self.addCleanup(repo.unlock)
    self.assertEqual({revision_id}, set(repo.all_revision_ids()))
    self.assertEqual({(revision_id,): ()}, repo.revisions.get_parent_map([(revision_id,)]))
    self.assertEqual({revision_id: (NULL_REVISION,)}, repo.get_parent_map([revision_id]))
    self.assertEqual({revision_id: (NULL_REVISION,)}, repo.get_graph().get_parent_map([revision_id]))
    other = self.make_repository('other')
    other.lock_read()
    self.addCleanup(other.unlock)
    self.assertEqual({revision_id: (NULL_REVISION,)}, repo.get_graph(other).get_parent_map([revision_id]))