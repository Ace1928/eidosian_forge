import unittest
from testtools.compat import _b
from subunit import content, content_type, details
def test_get_message(self):
    parser = details.SimpleDetailsParser(None)
    self.assertEqual(_b(''), parser.get_message())