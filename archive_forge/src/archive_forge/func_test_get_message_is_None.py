import unittest
from testtools.compat import _b
from subunit import content, content_type, details
def test_get_message_is_None(self):
    parser = details.MultipartDetailsParser(None)
    self.assertEqual(None, parser.get_message())