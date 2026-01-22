from breezy import gpg, tests
from breezy.bzr.testament import Testament
from breezy.controldir import ControlDir
def test_resign(self):
    wt, [a, b, c] = self.setup_tree()
    repo = wt.branch.repository
    self.monkey_patch_gpg()
    self.run_bzr('re-sign -r revid:%s' % a.decode('utf-8'))
    self.assertEqualSignature(repo, a)
    self.run_bzr('re-sign %s' % b.decode('utf-8'))
    self.assertEqualSignature(repo, b)