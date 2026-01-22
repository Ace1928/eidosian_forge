from breezy import errors, gpg
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventory, versionedfile, vf_repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_add_revision_inventory_sha1(self):
    inv = inventory.Inventory(revision_id=b'A')
    inv.root.revision = b'A'
    inv.root.file_id = b'fixed-root'
    reference_repo = self.make_repository('reference_repo')
    reference_repo.lock_write()
    reference_repo.start_write_group()
    inv_sha1 = reference_repo.add_inventory(b'A', inv, [])
    reference_repo.abort_write_group()
    reference_repo.unlock()
    repo = self.make_repository('repo')
    repo.lock_write()
    repo.start_write_group()
    root_id = inv.root.file_id
    repo.texts.add_lines((b'fixed-root', b'A'), [], [])
    repo.add_revision(b'A', _mod_revision.Revision(b'A', committer='B', timestamp=0, timezone=0, message='C'), inv=inv)
    repo.commit_write_group()
    repo.unlock()
    repo.lock_read()
    self.assertEqual(inv_sha1, repo.get_revision(b'A').inventory_sha1)
    repo.unlock()