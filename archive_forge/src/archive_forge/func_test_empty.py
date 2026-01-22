import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_empty(self):
    stream = XML('<elem></elem>') | EmptyTagFilter()
    self.assertEqual([EmptyTagFilter.EMPTY], [ev[0] for ev in stream])