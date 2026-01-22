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
def test_fetch_missing_basis_text(self):
    """If fetching a delta, we should die if a basis is not present."""
    if not self.repository_format.supports_full_versioned_files:
        raise TestNotApplicable('Need full versioned files support')
    if not self.repository_format_to.supports_full_versioned_files:
        raise TestNotApplicable('Need full versioned files support')
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/a'])
    tree.add(['a'])
    rev1 = tree.commit('one')
    self.build_tree_contents([('tree/a', b'new contents\n')])
    rev2 = tree.commit('two')
    to_repo = self.make_to_repository('to_repo')
    with to_repo.lock_write():
        with WriteGroup(to_repo, suppress_errors=True):
            inv = tree.branch.repository.get_inventory(rev1)
            to_repo.add_inventory(rev1, inv, [])
            rev = tree.branch.repository.get_revision(rev1)
            to_repo.add_revision(rev1, rev, inv=inv)
            self.disable_commit_write_group_paranoia(to_repo)
    try:
        to_repo.fetch(tree.branch.repository, rev2)
    except (errors.BzrCheckError, errors.RevisionNotPresent) as e:
        self.assertRaises((errors.NoSuchRevision, errors.RevisionNotPresent), to_repo.revision_tree, rev2)
    else:
        with to_repo.lock_read():
            rt = to_repo.revision_tree(rev2)
            self.assertEqual(b'new contents\n', rt.get_file_text('a'))