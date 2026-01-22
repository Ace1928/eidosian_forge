from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_parse_delete(self):
    lines = root_only_lines
    lines += b'/old-file\x00None\x00deleted-id\x00\x00null:\x00deleted\x00\x00\n'
    deserializer = inventory_delta.InventoryDeltaDeserializer()
    parse_result = deserializer.parse_text_bytes(osutils.split_lines(lines))
    delta = parse_result[4]
    self.assertEqual(('old-file', None, b'deleted-id', None), delta[-1])