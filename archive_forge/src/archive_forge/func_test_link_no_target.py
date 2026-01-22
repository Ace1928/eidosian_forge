from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_link_no_target(self):
    entry = inventory.make_entry('symlink', 'a link', None)
    self.assertRaises(InventoryDeltaError, inventory_delta._link_content, entry)