import unittest
from testtools.compat import _b
from subunit import content, content_type, details
def test_get_details_success(self):
    parser = details.SimpleDetailsParser(None)
    expected = {}
    expected['message'] = content.Content(content_type.ContentType('text', 'plain'), lambda: [_b('')])
    found = parser.get_details('success')
    self.assertEqual(expected, found)