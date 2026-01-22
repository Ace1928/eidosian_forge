from breezy import gpg, tests
def test_sign_my_commits(self):
    wt = self.setup_tree()
    repo = wt.branch.repository
    self.monkey_patch_gpg()
    self.assertUnsigned(repo, b'A')
    self.assertUnsigned(repo, b'B')
    self.assertUnsigned(repo, b'C')
    self.assertUnsigned(repo, b'D')
    self.run_bzr('sign-my-commits')
    self.assertSigned(repo, b'A')
    self.assertSigned(repo, b'B')
    self.assertSigned(repo, b'C')
    self.assertUnsigned(repo, b'D')