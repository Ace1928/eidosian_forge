from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils
from ... import revision as _mod_revision
from ...bzr import inventory
from ...bzr.inventorytree import InventoryTreeChange
def init_inventory(self, revision_id):
    """Generate an inventory for a parentless revision."""
    if self._supports_chks:
        inv = self._init_chk_inventory(revision_id, inventory.ROOT_ID)
    else:
        inv = inventory.Inventory(revision_id=revision_id)
        if self.expects_rich_root():
            inv.root.revision = revision_id
    return inv