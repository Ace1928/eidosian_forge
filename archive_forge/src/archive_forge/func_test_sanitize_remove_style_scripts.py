import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
def test_sanitize_remove_style_scripts(self):
    sanitizer = StyleSanitizer()
    html = HTML(u'<DIV STYLE=\'background: url(javascript:alert("foo"))\'>')
    self.assertEqual('<div/>', (html | sanitizer).render())
    html = HTML(u'<DIV STYLE=\'background: url(&#1;javascript:alert("foo"))\'>')
    self.assertEqual('<div/>', (html | sanitizer).render())
    html = HTML(u'<DIV STYLE=\'background: url("javascript:alert(foo)")\'>')
    self.assertEqual('<div/>', (html | sanitizer).render())
    html = HTML(u'<DIV STYLE=\'width: expression(alert("foo"));\'>')
    self.assertEqual('<div/>', (html | sanitizer).render())
    html = HTML(u'<DIV STYLE=\'width: e/**/xpression(alert("foo"));\'>')
    self.assertEqual('<div/>', (html | sanitizer).render())
    html = HTML(u'<DIV STYLE=\'background: url(javascript:alert("foo"));color: #fff\'>')
    self.assertEqual('<div style="color: #fff"/>', (html | sanitizer).render())
    html = HTML(u'<DIV STYLE=\'background: \\75rl(javascript:alert("foo"))\'>')
    self.assertEqual('<div/>', (html | sanitizer).render())
    html = HTML(u'<DIV STYLE=\'background: \\000075rl(javascript:alert("foo"))\'>')
    self.assertEqual('<div/>', (html | sanitizer).render())
    html = HTML(u'<DIV STYLE=\'background: \\75 rl(javascript:alert("foo"))\'>')
    self.assertEqual('<div/>', (html | sanitizer).render())
    html = HTML(u'<DIV STYLE=\'background: \\000075 rl(javascript:alert("foo"))\'>')
    self.assertEqual('<div/>', (html | sanitizer).render())
    html = HTML(u'<DIV STYLE=\'background: \\000075\r\nrl(javascript:alert("foo"))\'>')
    self.assertEqual('<div/>', (html | sanitizer).render())