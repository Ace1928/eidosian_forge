from breezy import branch, controldir, errors, tests
from breezy.tests import script
def test_unbinding(self):
    base_tree, child_tree = self.create_branches()
    self.build_tree_contents([('base/a', b'new base contents\n'), ('child/b', b'new b child contents\n')])
    base_tree.commit(message='base')
    self.check_revno(2, 'base')
    self.check_revno(1, 'child')
    self.run_bzr('commit -m child', retcode=3, working_dir='child')
    self.check_revno(1, 'child')
    self.run_bzr('unbind', working_dir='child')
    child_tree = child_tree.controldir.open_workingtree()
    child_tree.commit(message='child')
    self.check_revno(2, 'child')