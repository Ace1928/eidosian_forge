import unittest
from genshi.core import Attrs, QName, Stream
from genshi.input import XMLParser, HTMLParser, ParseError, ET
from genshi.compat import StringIO, BytesIO
from genshi.tests.test_utils import doctest_suite
from xml.etree import ElementTree
def test_out_of_order_tags1(self):
    text = u'<span><b>Foobar</span></b>'
    events = list(HTMLParser(StringIO(text)))
    self.assertEqual(5, len(events))
    self.assertEqual((Stream.START, ('span', ())), events[0][:2])
    self.assertEqual((Stream.START, ('b', ())), events[1][:2])
    self.assertEqual((Stream.TEXT, 'Foobar'), events[2][:2])
    self.assertEqual((Stream.END, 'b'), events[3][:2])
    self.assertEqual((Stream.END, 'span'), events[4][:2])