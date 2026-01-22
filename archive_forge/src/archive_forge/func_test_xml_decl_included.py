import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_xml_decl_included(self):
    stream = Stream([(Stream.XML_DECL, ('1.0', None, -1), (None, -1, -1))])
    output = stream.render(XHTMLSerializer, doctype='xhtml', drop_xml_decl=False, encoding=None)
    self.assertEqual('<?xml version="1.0"?>\n<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n', output)