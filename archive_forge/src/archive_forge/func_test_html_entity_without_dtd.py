import unittest
from genshi.core import Attrs, QName, Stream
from genshi.input import XMLParser, HTMLParser, ParseError, ET
from genshi.compat import StringIO, BytesIO
from genshi.tests.test_utils import doctest_suite
from xml.etree import ElementTree
def test_html_entity_without_dtd(self):
    text = '<html>&nbsp;</html>'
    events = list(XMLParser(StringIO(text)))
    kind, data, pos = events[1]
    self.assertEqual(Stream.TEXT, kind)
    self.assertEqual(u'\xa0', data)