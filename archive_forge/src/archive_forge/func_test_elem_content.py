import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_elem_content(self):
    stream = XML('<elem><sub /><sub /></elem>') | EmptyTagFilter()
    self.assertEqual([Stream.START, EmptyTagFilter.EMPTY, EmptyTagFilter.EMPTY, Stream.END], [ev[0] for ev in stream])