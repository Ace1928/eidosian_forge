import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_namespace_context(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n                                       xmlns:x="http://www.example.org/">\n          <div py:match="x:foo">Foo</div>\n          <foo xmlns="http://www.example.org/"/>\n        </html>')
    self.assertEqual('<html xmlns:x="http://www.example.org/">\n          <div>Foo</div>\n        </html>', tmpl.generate().render(encoding=None))