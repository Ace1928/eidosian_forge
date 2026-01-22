import os
from breezy import merge, tests, transform, workingtree
def test_revert_new(self):
    """Only locally-changed new files should be preserved when reverting

        When a file isn't present in revert's target tree:
        If a file hasn't been committed, revert should unversion it, but not
        delete it.
        If a file has local changes, revert should unversion it, but not
        delete it.
        If a file has no changes from the last commit, revert should delete it.
        If a file has changes due to a merge, revert should delete it.
        """
    tree = self.make_branch_and_tree('tree')
    tree.commit('empty tree')
    merge_target = tree.controldir.sprout('merge_target').open_workingtree()
    self.build_tree(['tree/new_file'])
    tree.add('new_file')
    basis_tree = tree.branch.repository.revision_tree(tree.last_revision())
    tree.revert()
    self.assertPathExists('tree/new_file')
    tree.add('new_file')
    tree.commit('add new_file')
    tree.revert(old_tree=basis_tree)
    self.assertPathDoesNotExist('tree/new_file')
    merge_target.merge_from_branch(tree.branch)
    self.assertPathExists('merge_target/new_file')
    merge_target.revert()
    self.assertPathDoesNotExist('merge_target/new_file')
    merge_target.merge_from_branch(tree.branch)
    self.assertPathExists('merge_target/new_file')
    self.build_tree_contents([('merge_target/new_file', b'new_contents')])
    merge_target.revert()
    self.assertPathExists('merge_target/new_file')