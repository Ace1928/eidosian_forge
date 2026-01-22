from breezy import branch, controldir, errors, tests
from breezy.tests import script
def test_commit_after_merge(self):
    base_tree, child_tree = self.create_branches()
    other_tree = child_tree.controldir.sprout('other').open_workingtree()
    other_branch = other_tree.branch
    self.build_tree_contents([('other/c', b'file c\n')])
    other_tree.add('c')
    other_tree.commit(message='adding c')
    new_rev_id = other_branch.last_revision()
    child_tree.merge_from_branch(other_branch)
    self.assertPathExists('child/c')
    self.assertEqual([new_rev_id], child_tree.get_parent_ids()[1:])
    self.assertTrue(child_tree.branch.repository.has_revision(new_rev_id))
    self.assertFalse(base_tree.branch.repository.has_revision(new_rev_id))
    self.run_bzr(['commit', '-m', 'merge other'], working_dir='child')
    self.check_revno(2, 'child')
    self.check_revno(2, 'base')
    self.assertTrue(base_tree.branch.repository.has_revision(new_rev_id))