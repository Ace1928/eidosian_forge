from dulwich.tests import TestCase
from ..line_ending import (
from ..objects import Blob
def test_convert_crlf_to_lf_no_op(self):
    self.assertEqual(convert_crlf_to_lf(b'foobar'), b'foobar')