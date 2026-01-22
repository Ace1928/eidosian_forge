import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def test_no_upstream_branch(self):
    self.run_bzr_error(['brz: ERROR: No upstream branch specified.\n'], 'rebase')