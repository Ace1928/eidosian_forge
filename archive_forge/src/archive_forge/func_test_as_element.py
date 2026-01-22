import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_as_element(self):
    tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <py:with vars="x = x * 2">${x}</py:with>\n        </div>')
    self.assertEqual('<div>\n          84\n        </div>', tmpl.generate(x=42).render(encoding=None))