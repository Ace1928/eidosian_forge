import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_sanitize_escape_text(self):
    html = HTML(u'<a href="#">fo&amp;</a>')
    self.assertEqual('<a href="#">fo&amp;</a>', (html | HTMLSanitizer()).render())
    html = HTML(u'<a href="#">&lt;foo&gt;</a>')
    self.assertEqual('<a href="#">&lt;foo&gt;</a>', (html | HTMLSanitizer()).render())