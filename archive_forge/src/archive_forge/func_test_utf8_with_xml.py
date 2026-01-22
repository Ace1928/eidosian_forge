from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_utf8_with_xml(self):
    utf8_str = b'\xc2\xb5\xc3\xa5&\xd8\xac'
    self.assertEqual(b'&#181;&#229;&amp;&#1580;', breezy.bzr.xml_serializer.encode_and_escape(utf8_str))