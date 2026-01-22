import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_non_ascii_str_committer_rejected(self):
    """Ensure an error is raised on a non-ascii byte string committer"""
    branch = self.make_branch('.')
    branch.repository.lock_write()
    self.addCleanup(branch.repository.unlock)
    self.assertRaises(UnicodeDecodeError, branch.repository.get_commit_builder, branch, [], branch.get_config_stack(), committer=b'Erik B\xe5gfors <erik@example.com>')