from breezy import branch, controldir, errors, tests
from breezy.tests import script
def test_pull_overwrite(self):
    child_tree = self.create_branches()[1]
    other_tree = child_tree.controldir.sprout('other').open_workingtree()
    self.build_tree_contents([('other/a', b'new contents\n')])
    other_tree.commit(message='changed a')
    self.check_revno(2, 'other')
    self.build_tree_contents([('other/a', b'new contents\nand then some\n')])
    other_tree.commit(message='another a')
    self.check_revno(3, 'other')
    self.build_tree_contents([('other/a', b'new contents\nand then some\nand some more\n')])
    other_tree.commit('yet another a')
    self.check_revno(4, 'other')
    self.build_tree_contents([('child/a', b'also changed a\n')])
    child_tree.commit(message='child modified a')
    self.check_revno(2, 'child')
    self.check_revno(2, 'base')
    self.run_bzr('pull --overwrite ../other', working_dir='child')
    self.check_revno(4, 'child')
    self.check_revno(4, 'base')