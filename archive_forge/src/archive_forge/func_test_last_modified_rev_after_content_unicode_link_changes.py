import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_last_modified_rev_after_content_unicode_link_changes(self):
    self.requireFeature(features.UnicodeFilenameFeature)
    self._test_last_mod_rev_after_content_link_changes('liሴnk', 'targ€t', 'n€wtarget')