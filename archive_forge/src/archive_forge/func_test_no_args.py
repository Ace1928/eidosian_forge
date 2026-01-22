from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_no_args(self):
    branch = self.make_branch('branch')
    self.run_bzr_error(['No target configuration specified'], 'reconfigure', working_dir='branch')