from breezy import branch, config, controldir, errors, osutils, tests
from breezy.tests.script import run_script
def test_break_lock_help(self):
    out, err = self.run_bzr('break-lock --help')
    self.assertEqual('', err)