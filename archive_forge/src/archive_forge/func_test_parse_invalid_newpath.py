from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_parse_invalid_newpath(self):
    """newpath must start with / if it is not None."""
    lines = empty_lines
    lines += b'None\x00bad\x00TREE_ROOT\x00\x00version\x00dir\n'
    deserializer = inventory_delta.InventoryDeltaDeserializer()
    err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, osutils.split_lines(lines))
    self.assertContainsRe(str(err), 'newpath invalid')