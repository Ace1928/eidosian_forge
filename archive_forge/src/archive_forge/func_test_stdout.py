from breezy import tests
from breezy.tests import features
def test_stdout(self):
    out, err = self.run_bzr('--lsprof rocks')
    self.assertContainsRe(out, 'CallCount')
    self.assertNotContainsRe(err, 'Profile data written to')