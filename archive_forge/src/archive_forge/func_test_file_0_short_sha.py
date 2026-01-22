from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_file_0_short_sha(self):
    file_entry = inventory.make_entry('file', 'a file', None, b'file-id')
    file_entry.text_sha1 = b''
    file_entry.text_size = 0
    self.assertEqual(b'file\x000\x00\x00', inventory_delta._file_content(file_entry))