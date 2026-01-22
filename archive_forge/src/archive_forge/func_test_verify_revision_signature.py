from breezy import errors, gpg, tests, urlutils
from breezy.bzr.testament import Testament
from breezy.repository import WriteGroup
from breezy.tests import per_repository
def test_verify_revision_signature(self):
    wt = self.make_branch_and_tree('.')
    a = wt.commit('base', allow_pointless=True)
    strategy = gpg.LoopbackGPGStrategy(None)
    repo = wt.branch.repository
    self.addCleanup(repo.lock_write().unlock)
    repo.start_write_group()
    repo.sign_revision(a, strategy)
    repo.commit_write_group()
    self.assertEqual(b'-----BEGIN PSEUDO-SIGNED CONTENT-----\n' + Testament.from_revision(repo, a).as_short_text() + b'-----END PSEUDO-SIGNED CONTENT-----\n', repo.get_signature_text(a))
    self.assertEqual((gpg.SIGNATURE_VALID, None), repo.verify_revision_signature(a, strategy))