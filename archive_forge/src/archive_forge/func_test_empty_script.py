import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_empty_script(self):
    text = '<script src="foo.js" />'
    output = XML(text).render(HTMLSerializer, encoding=None)
    self.assertEqual('<script src="foo.js"></script>', output)