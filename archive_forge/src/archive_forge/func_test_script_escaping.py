import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_script_escaping(self):
    text = '<script>if (1 &lt; 2) { alert("Doh"); }</script>'
    output = XML(text).render(HTMLSerializer, encoding=None)
    self.assertEqual('<script>if (1 < 2) { alert("Doh"); }</script>', output)