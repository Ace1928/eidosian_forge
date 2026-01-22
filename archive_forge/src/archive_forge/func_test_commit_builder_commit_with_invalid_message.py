import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_commit_builder_commit_with_invalid_message(self):
    branch = self.make_branch('.')
    branch.repository.lock_write()
    self.addCleanup(branch.repository.unlock)
    builder = branch.repository.get_commit_builder(branch, [], branch.get_config_stack())
    self.addCleanup(branch.repository.abort_write_group)
    self.assertRaises(ValueError, builder.commit, 'Invalid\r\ncommit message\r\n')