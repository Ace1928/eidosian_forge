from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_revision_text_v8(self):
    """Pack revision to XML v8"""
    rev = breezy.bzr.xml8.serializer_v8.read_revision_from_string(_expected_rev_v8)
    serialized = breezy.bzr.xml8.serializer_v8.write_revision_to_lines(rev)
    self.assertEqualDiff(b''.join(serialized), _expected_rev_v8)