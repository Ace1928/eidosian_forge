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
def test_fetch_missing_text_other_location_fails(self):
    if not self.repository_format.supports_full_versioned_files:
        raise TestNotApplicable('Need full versioned files')
    source_tree = self.make_branch_and_tree('source')
    source = source_tree.branch.repository
    target = self.make_to_repository('target')
    self.build_tree(['source/id'])
    source_tree.add(['id'], ids=[b'id'])
    source_tree.commit('a', rev_id=b'a')
    inv = source.get_inventory(b'a')
    source.lock_write()
    self.addCleanup(source.unlock)
    source.start_write_group()
    inv.get_entry(b'id').revision = b'b'
    inv.revision_id = b'b'
    sha1 = source.add_inventory(b'b', inv, [b'a'])
    rev = Revision(timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', message='Message', inventory_sha1=sha1, revision_id=b'b')
    rev.parent_ids = [b'a']
    source.add_revision(b'b', rev)
    self.disable_commit_write_group_paranoia(source)
    source.commit_write_group()
    try:
        self.assertRaises(errors.RevisionNotPresent, target.fetch, source)
    except errors.NoRoundtrippingSupport:
        raise TestNotApplicable('roundtripping not supported')
    self.assertFalse(target.has_revision(b'b'))