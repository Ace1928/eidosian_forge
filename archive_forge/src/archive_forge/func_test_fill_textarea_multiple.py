import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_fill_textarea_multiple(self):
    html = HTML(u'<form><p>\n          <textarea name="foo"></textarea>\n          <textarea name="bar"></textarea>\n        </p></form>') | HTMLFormFiller(data={'foo': 'Some text'})
    self.assertEqual('<form><p>\n          <textarea name="foo">Some text</textarea>\n          <textarea name="bar"/>\n        </p></form>', html.render())