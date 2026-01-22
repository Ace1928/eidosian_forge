from breezy import gpg, tests
def test_verify_signatures_acceptable_key(self):
    wt = self.setup_tree()
    self.monkey_patch_gpg()
    self.run_bzr('sign-my-commits')
    out = self.run_bzr(['verify-signatures', '--acceptable-keys=foo,bar'], retcode=1)
    self.assertEqual(('4 commits with valid signatures\n0 commits with key now expired\n0 commits with unknown keys\n0 commits not valid\n1 commit not signed\n', ''), out)