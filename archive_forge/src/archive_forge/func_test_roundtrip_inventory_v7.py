from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_roundtrip_inventory_v7(self):
    inv = self.get_sample_inventory()
    inv.add(inventory.TreeReference(b'nested-id', 'nested', b'tree-root-321', b'rev_outer', b'rev_inner'))
    lines = xml7.serializer_v7.write_inventory_to_lines(inv)
    self.assertEqualDiff(_expected_inv_v7, b''.join(lines))
    inv2 = xml7.serializer_v7.read_inventory_from_lines(lines)
    self.assertEqual(5, len(inv2))
    for path, ie in inv.iter_entries():
        self.assertEqual(ie, inv2.get_entry(ie.file_id))