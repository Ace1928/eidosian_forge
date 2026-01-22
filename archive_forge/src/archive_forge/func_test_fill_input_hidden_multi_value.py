import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_fill_input_hidden_multi_value(self):
    html = HTML(u'<form><p>\n          <input type="hidden" name="foo" />\n        </p></form>') | HTMLFormFiller(data={'foo': ['bar']})
    self.assertEqual('<form><p>\n          <input type="hidden" name="foo" value="bar"/>\n        </p></form>', html.render())