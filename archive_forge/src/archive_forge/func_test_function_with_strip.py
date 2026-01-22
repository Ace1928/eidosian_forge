import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_function_with_strip(self):
    """
        Verify that a named template function with a strip directive actually
        strips of the outer element.
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:def="echo(what)" py:strip="">\n            <b>${what}</b>\n          </div>\n          ${echo(\'foo\')}\n        </doc>')
    self.assertEqual('<doc>\n            <b>foo</b>\n        </doc>', tmpl.generate().render(encoding=None))