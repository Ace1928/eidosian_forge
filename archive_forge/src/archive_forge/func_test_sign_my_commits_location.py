from breezy import gpg, tests
def test_sign_my_commits_location(self):
    wt = self.setup_tree('other')
    repo = wt.branch.repository
    self.monkey_patch_gpg()
    self.run_bzr('sign-my-commits other')
    self.assertSigned(repo, b'A')
    self.assertSigned(repo, b'B')
    self.assertSigned(repo, b'C')
    self.assertUnsigned(repo, b'D')