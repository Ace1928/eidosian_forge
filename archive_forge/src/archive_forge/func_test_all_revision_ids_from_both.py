from breezy.tests.per_repository_reference import \
def test_all_revision_ids_from_both(self):
    tree = self.make_branch_and_tree('spare')
    revid = tree.commit('one')
    base_tree = self.make_branch_and_tree('base')
    revid2 = base_tree.commit('two')
    repo = self.make_referring('referring', base_tree.branch.repository)
    repo.fetch(tree.branch.repository, revid)
    self.assertEqual({revid, revid2}, set(repo.all_revision_ids()))