import doctest
import unittest
from genshi.builder import Element, tag
from genshi.core import Attrs, Markup, Stream
from genshi.input import XML
def test_duplicate_attributes(self):
    link = tag.a(href='#1', href_='#1')('Bar')
    events = list(link.generate())
    self.assertEqual((Stream.START, ('a', Attrs([('href', '#1')])), (None, -1, -1)), events[0])
    self.assertEqual((Stream.TEXT, 'Bar', (None, -1, -1)), events[1])
    self.assertEqual((Stream.END, 'a', (None, -1, -1)), events[2])