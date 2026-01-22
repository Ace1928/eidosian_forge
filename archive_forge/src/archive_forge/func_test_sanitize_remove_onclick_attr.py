import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_sanitize_remove_onclick_attr(self):
    html = HTML(u'<div onclick=\'alert("foo")\' />')
    self.assertEqual('<div/>', (html | HTMLSanitizer()).render())