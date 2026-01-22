import unittest
from testtools.compat import _b
from subunit import content, content_type, details
def test_get_details_skip(self):
    parser = details.SimpleDetailsParser(None)
    expected = {}
    expected['reason'] = content.Content(content_type.ContentType('text', 'plain'), lambda: [_b('')])
    found = parser.get_details('skip')
    self.assertEqual(expected, found)