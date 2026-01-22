import unittest
from testtools.compat import _b
from subunit import content, content_type, details
def test_get_details(self):
    parser = details.MultipartDetailsParser(None)
    self.assertEqual({}, parser.get_details())