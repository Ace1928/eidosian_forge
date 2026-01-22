from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_unknown_kind_errors(self):
    old_inv = Inventory(None)
    new_inv = Inventory(None)
    root = new_inv.make_entry('directory', '', None, b'my-rich-root-id')
    root.revision = b'changed'
    new_inv.add(root)

    class StrangeInventoryEntry(inventory.InventoryEntry):
        kind = 'strange'
    non_root = StrangeInventoryEntry(b'id', 'foo', root.file_id)
    non_root.revision = b'changed'
    new_inv.add(non_root)
    delta = new_inv._make_delta(old_inv)
    serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=True, tree_references=True)
    err = self.assertRaises(KeyError, serializer.delta_to_lines, NULL_REVISION, b'entry-version', delta)
    self.assertEqual(('strange',), err.args)