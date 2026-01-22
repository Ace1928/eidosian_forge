import sys
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy import tests, transform
from breezy.bzr import inventory, remote
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_file_ids_include_ghosts(self):
    b = self.create_branch_with_ghost_text()
    repo = b.repository
    self.assertEqual({b'a-file-id': {b'ghost-id'}}, repo.fileids_altered_by_revision_ids([b'B-id']))