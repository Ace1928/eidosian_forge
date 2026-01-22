import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_last_modified_file_dir(self):
    if not self.repository_format.supports_versioned_directories:
        raise tests.TestNotApplicable('format does not support versioned directories')
    self._check_kind_change(self.make_file, self.make_dir)