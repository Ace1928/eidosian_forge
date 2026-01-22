from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_wrong_format_v7(self):
    """Can't accidentally open a file with wrong serializer"""
    s_v6 = breezy.bzr.xml6.serializer_v6
    s_v7 = xml7.serializer_v7
    self.assertRaises(serializer.UnexpectedInventoryFormat, s_v7.read_inventory_from_lines, breezy.osutils.split_lines(_expected_inv_v5))
    self.assertRaises(serializer.UnexpectedInventoryFormat, s_v6.read_inventory_from_lines, breezy.osutils.split_lines(_expected_inv_v7))