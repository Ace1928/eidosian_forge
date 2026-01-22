from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_roundtrip_inventory_v8(self):
    inv = self.get_sample_inventory()
    lines = xml8.serializer_v8.write_inventory_to_lines(inv)
    inv2 = xml8.serializer_v8.read_inventory_from_lines(lines)
    self.assertEqual(4, len(inv2))
    for path, ie in inv.iter_entries():
        self.assertEqual(ie, inv2.get_entry(ie.file_id))