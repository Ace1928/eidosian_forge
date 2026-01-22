import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_doctype_in_stream_no_sysid(self):
    stream = Stream([(Stream.DOCTYPE, ('html', '-//W3C//DTD HTML 4.01//EN', None), (None, -1, -1))])
    output = stream.render(XMLSerializer, encoding=None)
    self.assertEqual('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN">\n', output)