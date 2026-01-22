import unittest
from genshi.core import Attrs, QName, Stream
from genshi.input import XMLParser, HTMLParser, ParseError, ET
from genshi.compat import StringIO, BytesIO
from genshi.tests.test_utils import doctest_suite
from xml.etree import ElementTree
def test_input_encoding_attribute(self):
    text = u'<div title="ö"></div>'.encode('iso-8859-1')
    events = list(HTMLParser(BytesIO(text), encoding='iso-8859-1'))
    kind, (tag, attrib), pos = events[0]
    self.assertEqual(Stream.START, kind)
    self.assertEqual(u'ö', attrib.get('title'))