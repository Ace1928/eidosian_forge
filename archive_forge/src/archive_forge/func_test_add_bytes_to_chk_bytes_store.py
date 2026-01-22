from breezy import errors, osutils, repository
from breezy.bzr import btree_index
from breezy.bzr.remote import RemoteRepository
from breezy.bzr.tests.per_repository_chk import TestCaseWithRepositoryCHK
from breezy.bzr.versionedfile import VersionedFiles
from breezy.tests import TestNotApplicable
def test_add_bytes_to_chk_bytes_store(self):
    repo = self.make_repository('.')
    with repo.lock_write(), repository.WriteGroup(repo):
        sha1, len, _ = repo.chk_bytes.add_lines((None,), None, [b'foo\n', b'bar\n'], random_id=True)
        self.assertEqual(b'4e48e2c9a3d2ca8a708cb0cc545700544efb5021', sha1)
        self.assertEqual({(b'sha1:4e48e2c9a3d2ca8a708cb0cc545700544efb5021',)}, repo.chk_bytes.keys())
    with repo.lock_read():
        self.assertEqual({(b'sha1:4e48e2c9a3d2ca8a708cb0cc545700544efb5021',)}, repo.chk_bytes.keys())
    repo = repo.controldir.open_repository()
    with repo.lock_read():
        self.assertEqual({(b'sha1:4e48e2c9a3d2ca8a708cb0cc545700544efb5021',)}, repo.chk_bytes.keys())