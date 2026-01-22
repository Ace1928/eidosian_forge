from breezy import gpg, tests
def test_sign_dry_run(self):
    wt = self.setup_tree()
    repo = wt.branch.repository
    self.monkey_patch_gpg()
    out = self.run_bzr('sign-my-commits --dry-run')[0]
    outlines = out.splitlines()
    self.assertEqual(5, len(outlines))
    self.assertEqual('Signed 4 revisions.', outlines[-1])
    self.assertUnsigned(repo, b'A')
    self.assertUnsigned(repo, b'B')
    self.assertUnsigned(repo, b'C')
    self.assertUnsigned(repo, b'D')
    self.assertUnsigned(repo, b'E')