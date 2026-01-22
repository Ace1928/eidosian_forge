from breezy import branch, controldir, errors, tests
from breezy.tests import script
def test_bind_diverged(self):
    base_tree, child_tree = self.create_branches()
    base_branch = base_tree.branch
    child_branch = child_tree.branch
    self.run_bzr('unbind', working_dir='child')
    child_tree = child_tree.controldir.open_workingtree()
    child_tree.commit(message='child', allow_pointless=True)
    self.check_revno(2, 'child')
    self.check_revno(1, 'base')
    base_tree.commit(message='base', allow_pointless=True)
    self.check_revno(2, 'base')
    self.run_bzr('bind ../base', working_dir='child')
    child_tree = child_tree.controldir.open_workingtree()
    child_tree.update()
    child_tree.commit(message='merged')
    self.check_revno(3, 'child')
    self.assertEqual(child_tree.branch.last_revision(), base_tree.branch.last_revision())