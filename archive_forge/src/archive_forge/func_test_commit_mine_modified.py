import os
from breezy import branch, builtins, errors
from breezy.tests import transport_util
def test_commit_mine_modified(self):
    self.start_logging_connections()
    commit = builtins.cmd_commit()
    os.chdir('local')
    commit.run(message='empty commit', unchanged=True)
    self.assertEqual(1, len(self.connections))