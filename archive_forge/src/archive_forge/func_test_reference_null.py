from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_reference_null(self):
    entry = inventory.make_entry('tree-reference', 'a tree', None)
    entry.reference_revision = NULL_REVISION
    self.assertEqual(b'tree\x00null:', inventory_delta._reference_content(entry))