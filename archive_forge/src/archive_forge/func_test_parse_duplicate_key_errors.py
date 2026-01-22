from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_parse_duplicate_key_errors(self):
    deserializer = inventory_delta.InventoryDeltaDeserializer()
    double_root_lines = b'format: bzr inventory delta v1 (bzr 1.14)\nparent: null:\nversion: null:\nversioned_root: true\ntree_references: true\nNone\x00/\x00an-id\x00\x00a@e\xc3\xa5ample.com--2004\x00dir\x00\x00\nNone\x00/\x00an-id\x00\x00a@e\xc3\xa5ample.com--2004\x00dir\x00\x00\n'
    err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, osutils.split_lines(double_root_lines))
    self.assertContainsRe(str(err), 'duplicate file id')