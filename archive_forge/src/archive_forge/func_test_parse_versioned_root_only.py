from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_parse_versioned_root_only(self):
    deserializer = inventory_delta.InventoryDeltaDeserializer()
    parse_result = deserializer.parse_text_bytes(osutils.split_lines(root_only_lines))
    expected_entry = inventory.make_entry('directory', '', None, b'an-id')
    expected_entry.revision = b'a@e\xc3\xa5ample.com--2004'
    self.assertEqual((b'null:', b'entry-version', True, True, [(None, '', b'an-id', expected_entry)]), parse_result)