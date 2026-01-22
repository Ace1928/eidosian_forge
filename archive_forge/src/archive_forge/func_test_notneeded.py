import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def test_notneeded(self):
    os.chdir('../feature')
    self.assertEqual('No revisions to rebase.\n', self.run_bzr('rebase ../main')[0])