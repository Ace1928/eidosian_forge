from breezy import branch, controldir, errors, tests
from breezy.tests import script
def test_bound_commit(self):
    child_tree = self.create_branches()[1]
    self.build_tree_contents([('child/a', b'new contents')])
    child_tree.commit(message='child')
    self.check_revno(2, 'child')
    self.check_revno(2, 'base')