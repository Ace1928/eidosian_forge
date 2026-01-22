import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_exec_in_replace(self):
    tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <p py:def="echo(greeting, name=\'world\')" class="message">\n            ${greeting}, ${name}!\n          </p>\n          <div py:replace="echo(\'hello\')"></div>\n        </div>')
    self.assertEqual('<div>\n          <p class="message">\n            hello, world!\n          </p>\n        </div>', tmpl.generate().render(encoding=None))