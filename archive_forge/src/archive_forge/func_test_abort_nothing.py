import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def test_abort_nothing(self):
    self.run_bzr_error(['brz: ERROR: No rebase to abort'], ['rebase-abort'])