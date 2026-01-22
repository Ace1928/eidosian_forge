import breezy
from breezy import errors
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.inventory import Inventory
from breezy.bzr.tests.per_repository_vf import (
from breezy.bzr.tests.per_repository_vf.helpers import \
from breezy.reconcile import Reconciler, reconcile
from breezy.revision import Revision
from breezy.tests import TestSkipped
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
from breezy.uncommit import uncommit
def test_text_from_ghost_revision(self):
    repo = self.make_repository('text-from-ghost')
    inv = Inventory(revision_id=b'final-revid')
    inv.root.revision = b'root-revid'
    ie = inv.add_path('bla', 'file', b'myfileid')
    ie.revision = b'ghostrevid'
    ie.text_size = 42
    ie.text_sha1 = b'bee68c8acd989f5f1765b4660695275948bf5c00'
    rev = breezy.revision.Revision(timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', message='Message', revision_id=b'final-revid')
    with repo.lock_write():
        repo.start_write_group()
        try:
            repo.add_revision(b'final-revid', rev, inv)
            try:
                repo.texts.add_lines((b'myfileid', b'ghostrevid'), ((b'myfileid', b'ghost-text-parent'),), [b'line1\n', b'line2\n'])
            except errors.RevisionNotPresent:
                raise TestSkipped('text ghost parents not supported')
            if repo.supports_rich_root():
                root_id = inv.root.file_id
                repo.texts.add_lines((inv.root.file_id, inv.root.revision), [], [])
        finally:
            repo.commit_write_group()
    repo.reconcile(thorough=True)