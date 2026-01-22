from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_unpack_inventory_5(self):
    """Unpack canned new-style inventory"""
    inp = BytesIO(_committed_inv_v5)
    inv = breezy.bzr.xml5.serializer_v5.read_inventory(inp)
    eq = self.assertEqual
    eq(len(inv), 4)
    ie = inv.get_entry(b'bar-20050824000535-6bc48cfad47ed134')
    eq(ie.kind, 'file')
    eq(ie.revision, b'mbp@foo-00')
    eq(ie.name, 'bar')
    eq(inv.get_entry(ie.parent_id).kind, 'directory')