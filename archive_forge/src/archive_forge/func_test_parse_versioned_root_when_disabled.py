from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_parse_versioned_root_when_disabled(self):
    deserializer = inventory_delta.InventoryDeltaDeserializer(allow_versioned_root=False)
    err = self.assertRaises(inventory_delta.IncompatibleInventoryDelta, deserializer.parse_text_bytes, osutils.split_lines(root_only_lines))
    self.assertEqual('versioned_root not allowed', str(err))