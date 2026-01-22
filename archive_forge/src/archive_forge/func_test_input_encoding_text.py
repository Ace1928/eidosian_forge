import unittest
from genshi.core import Attrs, QName, Stream
from genshi.input import XMLParser, HTMLParser, ParseError, ET
from genshi.compat import StringIO, BytesIO
from genshi.tests.test_utils import doctest_suite
from xml.etree import ElementTree
def test_input_encoding_text(self):
    text = u'<div>ö</div>'.encode('iso-8859-1')
    events = list(HTMLParser(BytesIO(text), encoding='iso-8859-1'))
    kind, data, pos = events[1]
    self.assertEqual(Stream.TEXT, kind)
    self.assertEqual(u'ö', data)