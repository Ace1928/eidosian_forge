from breezy import branch, config, controldir, errors, osutils, tests
from breezy.tests.script import run_script
def test_break_lock_no_interaction(self):
    """With --force, the user isn't asked for confirmation"""
    self.master_branch.lock_write()
    run_script(self, '\n        $ brz break-lock --force master-repo/master-branch\n        Broke lock ...master-branch/.bzr/...\n        ')
    self.assertRaises(errors.LockBroken, self.master_branch.unlock)