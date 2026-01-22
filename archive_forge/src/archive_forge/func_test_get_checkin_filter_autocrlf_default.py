from dulwich.tests import TestCase
from ..line_ending import (
from ..objects import Blob
def test_get_checkin_filter_autocrlf_default(self):
    checkin_filter = get_checkin_filter_autocrlf(b'false')
    self.assertEqual(checkin_filter, None)