from dulwich.tests import TestCase
from ..line_ending import (
from ..objects import Blob
def test_get_checkout_filter_autocrlf_true(self):
    checkout_filter = get_checkout_filter_autocrlf(b'true')
    self.assertEqual(checkout_filter, convert_lf_to_crlf)