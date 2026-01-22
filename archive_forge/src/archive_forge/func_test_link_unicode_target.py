from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_link_unicode_target(self):
    entry = inventory.make_entry('symlink', 'a link', None)
    entry.symlink_target = b' \xc3\xa5'.decode('utf8')
    self.assertEqual(b'link\x00 \xc3\xa5', inventory_delta._link_content(entry))