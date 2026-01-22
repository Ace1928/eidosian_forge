import sys
from breezy import errors, osutils, repository
from breezy.bzr import inventory, versionedfile
from breezy.bzr.vf_search import SearchResult
from breezy.errors import NoSuchRevision
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable
from breezy.tests.per_interrepository import TestCaseWithInterRepository
from breezy.tests.per_interrepository.test_interrepository import \
def test_fetch_inconsistent_last_changed_entries(self):
    """If an inventory has odd data we should still get what it references.

        This test tests that we do fetch a file text created in a revision not
        being fetched, but referenced from the revision we are fetching when the
        adjacent revisions to the one being fetched do not reference that text.
        """
    if not self.repository_format.supports_full_versioned_files:
        raise TestNotApplicable('Need full versioned files')
    tree = self.make_branch_and_tree('source')
    revid = tree.commit('old')
    to_repo = self.make_to_repository('to_repo')
    try:
        to_repo.fetch(tree.branch.repository, revid)
    except errors.NoRoundtrippingSupport:
        raise TestNotApplicable('roundtripping not supported')
    source = tree.branch.repository
    source.lock_write()
    self.addCleanup(source.unlock)
    with WriteGroup(source):
        source.texts.insert_record_stream([versionedfile.FulltextContentFactory((b'foo', revid), (), None, b'contents')])
        basis = source.revision_tree(revid)
        parent_id = basis.path2id('')
        entry = inventory.make_entry('file', 'foo-path', parent_id, b'foo')
        entry.revision = revid
        entry.text_size = len('contents')
        entry.text_sha1 = osutils.sha_string(b'contents')
        inv_sha1, _ = source.add_inventory_by_delta(revid, [(None, 'foo-path', b'foo', entry)], b'new', [revid])
        rev = Revision(timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', message='Message', inventory_sha1=inv_sha1, revision_id=b'new', parent_ids=[revid])
        source.add_revision(rev.revision_id, rev)
    to_repo.fetch(source, b'new')
    to_repo.lock_read()
    self.addCleanup(to_repo.unlock)
    self.assertEqual(b'contents', next(to_repo.texts.get_record_stream([(b'foo', revid)], 'unordered', True)).get_bytes_as('fulltext'))