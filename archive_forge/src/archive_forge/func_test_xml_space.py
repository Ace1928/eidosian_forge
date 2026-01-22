import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_xml_space(self):
    text = '<foo xml:space="preserve"> Do not mess  \n\n with me </foo>'
    output = XML(text).render(HTMLSerializer, encoding=None)
    self.assertEqual('<foo> Do not mess  \n\n with me </foo>', output)