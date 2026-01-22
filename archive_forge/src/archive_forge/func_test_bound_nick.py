import breezy
from breezy import branch, osutils, tests
def test_bound_nick(self):
    """Bind should not update implicit nick."""
    base = self.make_branch_and_tree('base')
    child = self.make_branch_and_tree('child')
    self.assertNick('child', working_dir='child', explicit=False)
    self.run_bzr('bind ../base', working_dir='child')
    self.assertNick(base.branch.nick, working_dir='child', explicit=False)