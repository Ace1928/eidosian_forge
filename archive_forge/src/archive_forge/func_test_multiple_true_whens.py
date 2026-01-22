import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_multiple_true_whens(self):
    """
        Verify that, if multiple `py:when` bodies match, only the first is
        output.
        """
    tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/" py:choose="">\n          <span py:when="1 == 1">1</span>\n          <span py:when="2 == 2">2</span>\n          <span py:when="3 == 3">3</span>\n        </div>')
    self.assertEqual('<div>\n          <span>1</span>\n        </div>', tmpl.generate().render(encoding=None))