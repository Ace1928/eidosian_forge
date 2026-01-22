from breezy.tests.per_repository_reference import \
def test_all_revision_ids_from_base(self):
    tree = self.make_branch_and_tree('base')
    revid = tree.commit('one')
    repo = self.make_referring('referring', tree.branch.repository)
    self.assertEqual({revid}, set(repo.all_revision_ids()))