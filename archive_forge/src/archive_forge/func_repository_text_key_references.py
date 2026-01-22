from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
def repository_text_key_references(self):
    result = {}
    if self.versioned_root:
        result.update({(b'TREE_ROOT', b'broken-revision-1-2'): True, (b'TREE_ROOT', b'broken-revision-2-1'): True, (b'TREE_ROOT', b'parent-1'): True, (b'TREE_ROOT', b'parent-2'): True})
    result.update({(b'a-file-id', b'broken-revision-1-2'): True, (b'a-file-id', b'broken-revision-2-1'): True, (b'a-file-id', b'parent-1'): True, (b'a-file-id', b'parent-2'): True})
    return result