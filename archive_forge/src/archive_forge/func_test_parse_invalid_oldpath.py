from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_parse_invalid_oldpath(self):
    """oldpath must start with / if it is not None."""
    lines = root_only_lines
    lines += b'bad\x00/new\x00file-id\x00\x00version\x00dir\n'
    deserializer = inventory_delta.InventoryDeltaDeserializer()
    err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, osutils.split_lines(lines))
    self.assertContainsRe(str(err), 'oldpath invalid')