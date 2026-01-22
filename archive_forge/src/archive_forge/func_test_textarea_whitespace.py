import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_textarea_whitespace(self):
    content = '\nHey there.  \n\n    I am indented.\n'
    stream = XML('<textarea name="foo">%s</textarea>' % content)
    output = stream.render(HTMLSerializer, encoding=None)
    self.assertEqual('<textarea name="foo">%s</textarea>' % content, output)