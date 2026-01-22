from breezy import branch, controldir, errors, tests
from breezy.tests import script
def test_bind_branch6(self):
    branch1 = self.make_branch('branch1', format='dirstate-tags')
    error = self.run_bzr('bind', retcode=3, working_dir='branch1')[1]
    self.assertEndsWith(error, 'No location supplied and no previous location known\n')