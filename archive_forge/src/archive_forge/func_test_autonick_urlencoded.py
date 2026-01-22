import breezy
from breezy import branch, osutils, tests
def test_autonick_urlencoded(self):
    self.make_branch_and_tree('!repo')
    self.assertNick('!repo', working_dir='!repo')