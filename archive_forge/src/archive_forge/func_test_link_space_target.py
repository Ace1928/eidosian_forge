from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_link_space_target(self):
    entry = inventory.make_entry('symlink', 'a link', None)
    entry.symlink_target = ' '
    self.assertEqual(b'link\x00 ', inventory_delta._link_content(entry))