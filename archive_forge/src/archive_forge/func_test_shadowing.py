import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_shadowing(self):
    tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          ${x}\n          <span py:with="x = x * 2" py:replace="x"/>\n          ${x}\n        </div>')
    self.assertEqual('<div>\n          42\n          84\n          42\n        </div>', tmpl.generate(x=42).render(encoding=None))