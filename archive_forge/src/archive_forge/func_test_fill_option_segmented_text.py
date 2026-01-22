import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_fill_option_segmented_text(self):
    html = MarkupTemplate(u'<form>\n          <select name="foo">\n            <option value="1">foo $x</option>\n          </select>\n        </form>').generate(x=1) | HTMLFormFiller(data={'foo': '1'})
    self.assertEqual(u'<form>\n          <select name="foo">\n            <option value="1" selected="selected">foo 1</option>\n          </select>\n        </form>', html.render())