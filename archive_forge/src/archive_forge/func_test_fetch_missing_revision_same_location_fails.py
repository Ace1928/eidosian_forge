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
def test_fetch_missing_revision_same_location_fails(self):
    repo_a = self.make_repository('.')
    repo_b = repository.Repository.open('.')
    self.assertRaises(errors.NoSuchRevision, repo_b.fetch, repo_a, revision_id=b'XXX')