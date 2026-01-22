import os
from io import StringIO
from .. import delta as _mod_delta
from .. import revision as _mod_revision
from .. import tests
from ..bzr.inventorytree import InventoryTreeChange
def test_modification(self):
    self.assertReport(' M  path', modified='modified')
    self.assertReport(' M* path', modified='modified', exe_change=True)