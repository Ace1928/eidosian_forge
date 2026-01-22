import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_fill_input_radio_single_value(self):
    html = HTML(u'<form><p>\n          <input type="radio" name="foo" value="1" />\n        </p></form>')
    self.assertEqual('<form><p>\n          <input type="radio" name="foo" value="1" checked="checked"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': '1'})).render())
    self.assertEqual('<form><p>\n          <input type="radio" name="foo" value="1"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': '2'})).render())