from breezy import errors, osutils, repository
from breezy.bzr import btree_index
from breezy.bzr.remote import RemoteRepository
from breezy.bzr.tests.per_repository_chk import TestCaseWithRepositoryCHK
from breezy.bzr.versionedfile import VersionedFiles
from breezy.tests import TestNotApplicable
def test_missing_text_record(self):
    """commit_write_group fails with BzrCheckError when a text is missing.
        """
    repo = self.make_repository('damaged-repo')
    b = self.make_branch_with_multiple_chk_nodes()
    src_repo = b.repository
    src_repo.lock_read()
    self.addCleanup(src_repo.unlock)
    all_texts = src_repo.texts.keys()
    all_texts.remove((b'file-%s-id' % (b'c' * 10000,), b'C-id'))
    repo.lock_write()
    repo.start_write_group()
    repo.chk_bytes.insert_record_stream(src_repo.chk_bytes.get_record_stream(src_repo.chk_bytes.keys(), 'unordered', True))
    repo.texts.insert_record_stream(src_repo.texts.get_record_stream(all_texts, 'unordered', True))
    repo.inventories.insert_record_stream(src_repo.inventories.get_record_stream([(b'B-id',), (b'C-id',)], 'unordered', True))
    repo.revisions.insert_record_stream(src_repo.revisions.get_record_stream([(b'C-id',)], 'unordered', True))
    repo.add_fallback_repository(b.repository)
    self.assertRaises(errors.BzrCheckError, repo.commit_write_group)
    reopened_repo = self.reopen_repo_and_resume_write_group(repo)
    self.assertRaises(errors.BzrCheckError, reopened_repo.commit_write_group)
    reopened_repo.abort_write_group()