from breezy import gpg, tests
from breezy.bzr.testament import Testament
from breezy.controldir import ControlDir
def test_resign_range(self):
    wt, [a, b, c] = self.setup_tree()
    repo = wt.branch.repository
    self.monkey_patch_gpg()
    self.run_bzr('re-sign -r 1..')
    self.assertEqualSignature(repo, a)
    self.assertEqualSignature(repo, b)
    self.assertEqualSignature(repo, c)