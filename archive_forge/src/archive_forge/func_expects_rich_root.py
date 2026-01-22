from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils
from ... import revision as _mod_revision
from ...bzr import inventory
from ...bzr.inventorytree import InventoryTreeChange
def expects_rich_root(self):
    """Does this store expect inventories with rich roots?"""
    return self.repo.supports_rich_root()