import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_sanitize_invalid_entity(self):
    html = HTML(u'&junk;')
    self.assertEqual('&amp;junk;', (html | HTMLSanitizer()).render())