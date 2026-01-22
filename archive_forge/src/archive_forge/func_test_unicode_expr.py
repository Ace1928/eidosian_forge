import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_unicode_expr(self):
    tmpl = MarkupTemplate(u'<div xmlns:py="http://genshi.edgewall.org/">\n          <span py:with="weeks=(u\'一\', u\'二\', u\'三\', u\'四\', u\'五\', u\'六\', u\'日\')">\n            $weeks\n          </span>\n        </div>')
    self.assertEqual(u'<div>\n          <span>\n            一二三四五六日\n          </span>\n        </div>', tmpl.generate().render(encoding=None))