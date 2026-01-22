import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_multiple_bound_namespaces(self):
    stream = Stream([(Stream.START, (QName('div'), Attrs()), (None, -1, -1)), (Stream.TEXT, '\n          ', (None, -1, -1)), (Stream.START_NS, ('x', 'http://example.org/'), (None, -1, -1)), (Stream.START, (QName('http://example.org/}p'), Attrs()), (None, -1, -1)), (Stream.END, QName('http://example.org/}p'), (None, -1, -1)), (Stream.END_NS, 'x', (None, -1, -1)), (Stream.TEXT, '\n          ', (None, -1, -1)), (Stream.START_NS, ('x', 'http://example.org/'), (None, -1, -1)), (Stream.START, (QName('http://example.org/}p'), Attrs()), (None, -1, -1)), (Stream.END, QName('http://example.org/}p'), (None, -1, -1)), (Stream.END_NS, 'x', (None, -1, -1)), (Stream.TEXT, '\n        ', (None, -1, -1)), (Stream.END, QName('div'), (None, -1, -1))])
    output = stream.render(XMLSerializer, encoding=None)
    self.assertEqual('<div>\n          <x:p xmlns:x="http://example.org/"/>\n          <x:p xmlns:x="http://example.org/"/>\n        </div>', output)