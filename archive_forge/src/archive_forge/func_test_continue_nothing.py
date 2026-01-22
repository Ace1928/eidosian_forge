import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def test_continue_nothing(self):
    self.run_bzr_error(['brz: ERROR: No rebase to continue'], ['rebase-continue'])