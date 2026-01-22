from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_unpack_inventory_5b(self):
    inv = breezy.bzr.xml5.serializer_v5.read_inventory_from_lines(breezy.osutils.split_lines(_inventory_v5b), revision_id=b'test-rev-id')
    self.assertEqual(b'a-rev-id', inv.root.revision)