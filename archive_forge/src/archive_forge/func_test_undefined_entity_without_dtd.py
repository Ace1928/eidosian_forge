import unittest
from genshi.core import Attrs, QName, Stream
from genshi.input import XMLParser, HTMLParser, ParseError, ET
from genshi.compat import StringIO, BytesIO
from genshi.tests.test_utils import doctest_suite
from xml.etree import ElementTree
def test_undefined_entity_without_dtd(self):
    text = '<html>&junk;</html>'
    events = XMLParser(StringIO(text))
    self.assertRaises(ParseError, list, events)