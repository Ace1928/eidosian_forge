import breezy
from breezy import branch, osutils, tests
def test_nick_directory(self):
    """Test --directory option"""
    self.make_branch_and_tree('me.dev')
    self.assertNick('me.dev', directory='me.dev')
    self.run_bzr(['nick', '-d', 'me.dev', 'moo'])
    self.assertNick('moo', directory='me.dev')