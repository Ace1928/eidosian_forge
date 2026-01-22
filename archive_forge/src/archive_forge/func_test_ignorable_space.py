import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_ignorable_space(self):
    text = '<foo> Mess  \n\n\n with me!  </foo>'
    output = XML(text).render(XMLSerializer, encoding=None)
    self.assertEqual('<foo> Mess\n with me!  </foo>', output)