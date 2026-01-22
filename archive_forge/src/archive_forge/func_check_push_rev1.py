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
def check_push_rev1(repo):
    self.assertRaises(NoSuchRevision, repo.get_revision, rev1)
    repo.fetch(tree_a.branch.repository, revision_id=NULL_REVISION)
    self.assertFalse(repo.has_revision(rev1))
    try:
        repo.fetch(tree_a.branch.repository)
    except errors.NoRoundtrippingSupport:
        raise TestNotApplicable('roundtripping not supported')
    rev = repo.get_revision(rev1)
    tree = repo.revision_tree(rev1)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    tree.get_file_text('foo')
    for path in tree.all_versioned_paths():
        if tree.kind(path) == 'file':
            with tree.get_file(path) as f:
                f.read()