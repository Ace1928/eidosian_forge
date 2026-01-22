from breezy import branch, controldir, errors, tests
from breezy.tests import script
def create_branches(self):
    base_tree = self.make_branch_and_tree('base')
    base_tree.lock_write()
    self.build_tree(['base/a', 'base/b'])
    base_tree.add(['a', 'b'])
    base_tree.commit('init')
    base_tree.unlock()
    child_tree = base_tree.branch.create_checkout('child')
    self.check_revno(1, 'child')
    d = controldir.ControlDir.open('child')
    self.assertNotEqual(None, d.open_branch().get_master_branch())
    return (base_tree, child_tree)