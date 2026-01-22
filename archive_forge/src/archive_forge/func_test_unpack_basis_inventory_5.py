from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_unpack_basis_inventory_5(self):
    """Unpack canned new-style inventory"""
    inv = breezy.bzr.xml5.serializer_v5.read_inventory_from_lines(breezy.osutils.split_lines(_basis_inv_v5))
    eq = self.assertEqual
    eq(len(inv), 4)
    eq(inv.revision_id, b'mbp@sourcefrog.net-20050905063503-43948f59fa127d92')
    ie = inv.get_entry(b'bar-20050824000535-6bc48cfad47ed134')
    eq(ie.kind, 'file')
    eq(ie.revision, b'mbp@foo-00')
    eq(ie.name, 'bar')
    eq(inv.get_entry(ie.parent_id).kind, 'directory')