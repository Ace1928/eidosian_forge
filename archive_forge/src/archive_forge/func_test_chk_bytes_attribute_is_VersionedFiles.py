from breezy import errors, osutils, repository
from breezy.bzr import btree_index
from breezy.bzr.remote import RemoteRepository
from breezy.bzr.tests.per_repository_chk import TestCaseWithRepositoryCHK
from breezy.bzr.versionedfile import VersionedFiles
from breezy.tests import TestNotApplicable
def test_chk_bytes_attribute_is_VersionedFiles(self):
    repo = self.make_repository('.')
    self.assertIsInstance(repo.chk_bytes, VersionedFiles)