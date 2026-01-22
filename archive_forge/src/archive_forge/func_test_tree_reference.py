from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_tree_reference(self):
    s_v5 = breezy.bzr.xml5.serializer_v5
    s_v6 = breezy.bzr.xml6.serializer_v6
    s_v7 = xml7.serializer_v7
    inv = Inventory(b'tree-root-321', revision_id=b'rev-outer')
    inv.root.revision = b'root-rev'
    inv.add(inventory.TreeReference(b'nested-id', 'nested', b'tree-root-321', b'rev-outer', b'rev-inner'))
    self.assertRaises(serializer.UnsupportedInventoryKind, s_v5.write_inventory_to_lines, inv)
    self.assertRaises(serializer.UnsupportedInventoryKind, s_v6.write_inventory_to_lines, inv)
    lines = s_v7.write_inventory_to_chunks(inv)
    inv2 = s_v7.read_inventory_from_lines(lines)
    self.assertEqual(b'tree-root-321', inv2.get_entry(b'nested-id').parent_id)
    self.assertEqual(b'rev-outer', inv2.get_entry(b'nested-id').revision)
    self.assertEqual(b'rev-inner', inv2.get_entry(b'nested-id').reference_revision)