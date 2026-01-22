import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_sanitize_remove_script_elem(self):
    html = HTML(u'<script>alert("Foo")</script>')
    self.assertEqual('', (html | HTMLSanitizer()).render())
    html = HTML(u'<SCRIPT SRC="http://example.com/"></SCRIPT>')
    self.assertEqual('', (html | HTMLSanitizer()).render())
    src = u'<SCR\x00IPT>alert("foo")</SCR\x00IPT>'
    self.assert_parse_error_or_equal('&lt;SCR\x00IPT&gt;alert("foo")', src, allow_strip=True)
    src = u'<SCRIPT&XYZ SRC="http://example.com/"></SCRIPT>'
    self.assert_parse_error_or_equal('&lt;SCRIPT&amp;XYZ; SRC="http://example.com/"&gt;', src, allow_strip=True)