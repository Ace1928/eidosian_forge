import unittest
from testtools.compat import _b
from subunit import content, content_type, details
def test_parts(self):
    parser = details.MultipartDetailsParser(None)
    parser.lineReceived(_b('Content-Type: text/plain\n'))
    parser.lineReceived(_b('something\n'))
    parser.lineReceived(_b('F\r\n'))
    parser.lineReceived(_b('serialised\n'))
    parser.lineReceived(_b('form0\r\n'))
    expected = {}
    expected['something'] = content.Content(content_type.ContentType('text', 'plain'), lambda: [_b('serialised\nform')])
    found = parser.get_details()
    self.assertEqual(expected.keys(), found.keys())
    self.assertEqual(expected['something'].content_type, found['something'].content_type)
    self.assertEqual(_b('').join(expected['something'].iter_bytes()), _b('').join(found['something'].iter_bytes()))