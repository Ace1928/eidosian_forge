import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_fill_input_checkbox_single_value_auto(self):
    html = HTML(u'<form><p>\n          <input type="checkbox" name="foo" />\n        </p></form>')
    self.assertEqual('<form><p>\n          <input type="checkbox" name="foo"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': ''})).render())
    self.assertEqual('<form><p>\n          <input type="checkbox" name="foo" checked="checked"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': 'on'})).render())