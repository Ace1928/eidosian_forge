import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_sanitize_property_name(self):
    html = HTML(u'<div style="display:none;border-left-color:red;user_defined:1;-moz-user-selct:-moz-all">prop</div>')
    self.assertEqual('<div style="display:none; border-left-color:red">prop</div>', six.text_type(html | StyleSanitizer()))