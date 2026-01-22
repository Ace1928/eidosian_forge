from breezy import errors, osutils, repository
from breezy.bzr import btree_index
from breezy.bzr.remote import RemoteRepository
from breezy.bzr.tests.per_repository_chk import TestCaseWithRepositoryCHK
from breezy.bzr.versionedfile import VersionedFiles
from breezy.tests import TestNotApplicable
def test_pack_preserves_chk_bytes_store(self):
    leaf_lines = [b'chkleaf:\n', b'0\n', b'1\n', b'0\n', b'\n']
    leaf_sha1 = osutils.sha_strings(leaf_lines)
    node_lines = [b'chknode:\n', b'0\n', b'1\n', b'1\n', b'foo\n', b'\x00sha1:%s\n' % (leaf_sha1,)]
    node_sha1 = osutils.sha_strings(node_lines)
    expected_set = {(b'sha1:' + leaf_sha1,), (b'sha1:' + node_sha1,)}
    repo = self.make_repository('.')
    with repo.lock_write():
        with repository.WriteGroup(repo):
            repo.chk_bytes.add_lines((None,), None, node_lines, random_id=True)
        with repository.WriteGroup(repo):
            repo.chk_bytes.add_lines((None,), None, leaf_lines, random_id=True)
        repo.pack()
        self.assertEqual(expected_set, repo.chk_bytes.keys())
    repo = repo.controldir.open_repository()
    with repo.lock_read():
        self.assertEqual(expected_set, repo.chk_bytes.keys())