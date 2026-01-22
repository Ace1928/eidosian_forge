from ...revision import Revision
from ..chk_serializer import chk_bencode_serializer
from . import TestCase
def test_unpack_revision_no_timezone(self):
    rev = chk_bencode_serializer.read_revision_from_string(_working_revision_bencode1_no_timezone)
    self.assertEqual(None, rev.timezone)