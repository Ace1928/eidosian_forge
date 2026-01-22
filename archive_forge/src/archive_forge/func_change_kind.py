import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def change_kind():
    if osutils.file_kind(path) == 'directory':
        osutils.rmtree(path)
    else:
        osutils.delete_any(path)
    make_after(path)