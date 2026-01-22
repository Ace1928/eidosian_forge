from breezy import errors, gpg, tests, urlutils
from breezy.bzr.testament import Testament
from breezy.repository import WriteGroup
from breezy.tests import per_repository
def test_clone_preserves_signatures(self):
    wt = self.make_branch_and_tree('source')
    a = wt.commit('A', allow_pointless=True)
    repo = wt.branch.repository
    repo.lock_write()
    repo.start_write_group()
    repo.sign_revision(a, gpg.LoopbackGPGStrategy(None))
    repo.commit_write_group()
    repo.unlock()
    self.build_tree(['target/'])
    d2 = repo.controldir.clone(urlutils.local_path_to_url('target'))
    self.assertEqual(repo.get_signature_text(a), d2.open_repository().get_signature_text(a))